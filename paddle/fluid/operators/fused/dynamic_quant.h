/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void DynamicQuantKernel(const T* input,
                                char4* output,
                                const T* in_scale,
                                const int m,
                                const int n) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (!check) return;
  float quant_in_scale = 1.0f / static_cast<float>(in_scale[m_id]);
  char4 tmp;
    constexpr int round_type = 1;
    constexpr float max_bound = 127.0f;
    constexpr float min_bound = -127.0f;
    tmp.x = quant_helper(input[m_id * n + n_id],
                         quant_in_scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.y = quant_helper(input[m_id * n + n_id + 1],
                         quant_in_scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.z = quant_helper(input[m_id * n + n_id + 2],
                         quant_in_scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.w = quant_helper(input[m_id * n + n_id + 3],
                         quant_in_scale,
                         round_type,
                         max_bound,
                         min_bound);
    output[(m_id * n + n_id) >> 2] = tmp;
}

template <typename T, int VecSize>
__global__ void DynamicDequantKernel(const int32_t* input,
                                  T* output,
                                  const int m,  // batch size
                                  const int n,  // hidden
                                  const T* quant_in_scale,
                                  const T* dequant_out_scale_data) {
  int numel = m * n;
  int stride = blockDim.x * gridDim.x * VecSize;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int raw_id = idx / n;
  int col_id = idx % n;

  phi::AlignedVector<int32_t, VecSize> in_vec;
  phi::AlignedVector<T, VecSize> out_scale_vec;
  phi::AlignedVector<T, VecSize> out_vec;

  float quant_in_scale_value = static_cast<float>(quant_in_scale[raw_id]) / 127.0f;



  for (; idx < numel; idx += stride) {
    phi::Load<int32_t, VecSize>(input + idx, &in_vec);
    phi::Load<T, VecSize>(dequant_out_scale_data + col_id, &out_scale_vec);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
        out_vec[i] = static_cast<T>(static_cast<float>(in_vec[i]) *
                                    quant_in_scale_value * static_cast<float>(out_scale_vec[i]));
    }

    phi::Store<T, VecSize>(out_vec, output + idx);
  }
}



template <typename T>
void LaunchDynamicQuantKernel(const T* input,
                                int8_t* output,
                                const T* in_scale,
                                const int m,
                                const int n,
                                gpuStream_t stream) {
  dim3 grid((n >> 2 + 31) / 32, (m + 31) / 32);
  dim3 block(32, 32);

  DynamicQuantKernel<<<grid, block, 0, stream>>>(input,
                                              (char4*)output,  // NOLINT
                                              in_scale,
                                              m,
                                              n);
}

template <typename T>
void LaunchDynamicDequantKernel(const int32_t* input,
                                  T* output,
                                  const int m,  // batch size
                                  const int n,  // hidden
                                  const T* quant_in_scale,
                                  const T* dequant_out_scale_data,
                                  gpuStream_t stream,
                                  GpuLaunchConfig* gpu_config) {
  VLOG(1) << "Launch dequantize_kernel";
  constexpr int DequantKernelVecSize = 128 / sizeof(T);
  DynamicDequantKernel<T, DequantKernelVecSize>
      <<<gpu_config->block_per_grid, gpu_config->thread_per_block, 0, stream>>>(
          input,
          output,
          m,
          n,
          quant_in_scale,
          dequant_out_scale_data);                                  
}


template <typename T>
void DyquantGemm(
                const phi::GPUContext& dev_ctx, 
                const phi::DenseTensor* weight,
                const phi::DenseTensor* input,
                phi::DenseTensor* output,
                const phi::DenseTensor* dequant_out_scale,
                std::string name,
                int m, int k, int n) {

    phi::DenseTensor quant_in_scale;
    quant_in_scale.Resize({m});
    dev_ctx.Alloc<T>(&quant_in_scale, m * sizeof(T));
                
    {
      VLOG(2) << "enter in max_kernel_launcher";
      phi::DenseTensor tmp;
      tmp.Resize(input->dims());
      dev_ctx.Alloc<T>(&tmp, input->numel() * sizeof(T));
      phi::AbsKernel<T>(dev_ctx, *input, &tmp);
      std::vector<int64_t> dims{-1};
      phi::MaxRawKernel<T>(dev_ctx, tmp, dims, false, false, &quant_in_scale);

      VLOG(2) << "end max_kernel_launcher";
    }
    // PrintMatrix(quant_in_scale.data<T>(), quant_in_scale.numel(), name + "_in_scale" + "_device_" + std::to_string(dev_ctx.GetPlace().GetDeviceId()));  
    
    phi::DenseTensor input_tmp, output_tmp;
    input_tmp.Resize({input->dims()});
    output_tmp.Resize({output->dims()});
    dev_ctx.Alloc<int8_t>(&input_tmp, input->numel() * sizeof(int8_t));
    dev_ctx.Alloc<int32_t>(&output_tmp, output->numel() * sizeof(int32_t));
    LaunchDynamicQuantKernel<T>(input->data<T>(),
                                input_tmp.data<int8_t>(),
                                quant_in_scale.data<T>(),
                                m,
                                k,
                                dev_ctx.stream());
    auto helper = std::make_unique<CublasLtHelper<int32_t>>(m, k, n, dev_ctx.cublaslt_handle());
    helper->GEMM(input_tmp.data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp.data<int32_t>(),
                      dev_ctx.stream());

    constexpr int DequantKernelVecSize = 128 / sizeof(T);
    auto gpu_config = std::make_unique<GpuLaunchConfig>(
        phi::backends::gpu::GetGpuLaunchConfig1D(
            dev_ctx, m * n, DequantKernelVecSize));
    LaunchDynamicDequantKernel<T>(output_tmp.data<int32_t>(),
                                  output->data<T>(),
                                  m,
                                  n,
                                  quant_in_scale.data<T>(),
                                  dequant_out_scale->data<T>(),
                                  dev_ctx.stream(),
                                  gpu_config.get());
}

}  // namespace operators
}  // namespace paddle