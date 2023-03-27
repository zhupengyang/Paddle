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

#include <iostream>
#include <vector>
#include "paddle/fluid/operators/fused/cublaslt.h"
#include "paddle/fluid/operators/fused/quant_dequant_kernel.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/fluid/operators/fused/datatype_traits.h"


#pragma once

namespace paddle {
namespace operators {

constexpr int32_t WARP_SIZE = 32; 
constexpr int32_t HALF_WARP = 16; 
constexpr float QUANT_MAX_BOUND = 127.0;
constexpr float QUANT_MIN_BOUND = -127.0;
// constexpr int KFP=128;

template<typename T>
struct MaxFunc{
  __device__ T operator()(T a, T b){
    return max(a, b); 
  }
}; 

template<>
struct MaxFunc<half>{
  __device__ half operator()(half a, half b){
#if __CUDA_ARCH__ >= 800
    return __hmax(a, b); 
#else
    return max(static_cast<float>(a), static_cast<float>(b));
#endif
  }
}; 


template<typename T>
struct AbsFunc{
  __device__ T operator()(T x){
    return abs(x); 
  }
}; 

template<>
struct AbsFunc<half>{
  __device__ half operator()(half x){
  #if __CUDA_ARCH__ >= 800
    return __habs(x); 
  #else
    return abs(static_cast<float>(x));
  #endif
  }
}; 

template<typename T>
struct QuantFunc{
  HOSTDEVICE int8_t operator()(T x, float inverse_range) {
    float tmp = static_cast<float>(x) * QUANT_MAX_BOUND *  inverse_range;
    tmp = round(tmp);
    if (tmp > QUANT_MAX_BOUND)
      tmp = QUANT_MAX_BOUND;
    else if (tmp < QUANT_MIN_BOUND)
      tmp = QUANT_MIN_BOUND;
    return static_cast<int8_t>(tmp);
  }
};

template<typename T>
struct DequantFunc{
  HOSTDEVICE T operator()(int8_t x, T scale) {
    return static_cast<T>(static_cast<float>(x) * static_cast<float>(scale));
  }
  HOSTDEVICE T operator()(int32_t x, T input_range, T weight_scale) {
    return static_cast<T>(static_cast<float>(x) * static_cast<float>(input_range) * static_cast<float>(weight_scale) / (127.0f));
  }
};

template <typename T, typename Vec, int VecSize>
__inline__ __device__ T LocalReduceMax(Vec& vec) {
  T local_max = static_cast<T>(0.0);
  #pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    local_max = vec[i] > local_max ?  vec[i] : local_max;
  }
  return local_max;
}

template <typename T>
__inline__ __device__ T WarpReduceAbsMax(T val, unsigned lane_mask) {
  #pragma unroll
  for (int mask = HALF_WARP; mask > 0; mask >>= 1){
    val = MaxFunc<T>()(val, __shfl_xor_sync(lane_mask, val, mask, WARP_SIZE));
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceAbsMax(T val, unsigned mask) {
  static __shared__ T smem[WARP_SIZE]; 
  int32_t lane_id = threadIdx.x & 0x1f; 
  int32_t warp_id = threadIdx.x >> 5; 
  val = WarpReduceAbsMax(val, mask); 
  if(lane_id == 0){
    smem[warp_id] = val; 
  }
  __syncthreads(); 
  T abs_max_val = (threadIdx.x < blockDim.x / WARP_SIZE) ? smem[threadIdx.x] : static_cast<T>(0.0f); 
  abs_max_val = WarpReduceAbsMax(abs_max_val, mask); 
  return abs_max_val; 
}

template<typename T, int VecSize>
__global__ void ReduceAbsMaxKernel(const T* x, const float threshold, const int32_t rows, const int32_t cols, 
                                   T* row_ranges, int32_t* outlier_idx){
  using InVec = phi::AlignedVector<T, VecSize>;

  InVec in_vec;
  InVec abs_max_vec;
  #pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    abs_max_vec[i] = 0.0;
  }


  T local_max_val = static_cast<T>(0.0); 
  for(int row_idx = blockIdx.x; row_idx < rows; row_idx += gridDim.x){
      for(int col_idx = threadIdx.x * VecSize; col_idx < cols; col_idx += blockDim.x * VecSize){
          int32_t linear_index = row_idx * cols + col_idx; 
          phi::Load<T, VecSize>(x + linear_index, &in_vec);
          #pragma unroll
          for (int i = 0; i < VecSize; ++i) {
            in_vec[i] = AbsFunc<T>()(in_vec[i]);
            if (in_vec[i] > static_cast<T>(threshold)) {
              int32_t index = col_idx + i;
              int32_t int_index = index / 32;
              int32_t inner_index = index % 32;
              // outlier_idx[int_index] |= (1 << inner_index);
              atomicOr(outlier_idx + int_index, (1 << inner_index));
              in_vec[i] = 0.0;
            }
            abs_max_vec[i] = MaxFunc<T>()(abs_max_vec[i], in_vec[i]);
          }
      }
      local_max_val = LocalReduceMax<T, InVec, VecSize>(abs_max_vec); 
      // __shared__ float inverse_row_max_val[1]; 
      T tmp_max_val = BlockReduceAbsMax<T>(local_max_val, 0xffffffff); 
      if(threadIdx.x == 0){
        row_ranges[row_idx] = tmp_max_val; 
        // inverse_row_max_val[0] = (1.0f / static_cast<float>(tmp_max_val));
      }
      // __syncthreads();

      // for(int col_idx = threadIdx.x * VecSize; col_idx < cols; col_idx += blockDim.x * VecSize){
      //     int32_t linear_index = row_idx * cols + col_idx; 
      //     phi::Load<T, VecSize>(x + linear_index, &in_vec);

      //   #pragma unroll
      //     for (int i = 0; i < VecSize; ++i) {
      //       if (in_vec[i] < static_cast<T>(threshold)) {
      //         out_vec[i] = QuantFunc<T>()(in_vec[i], inverse_row_max_val[0]);
      //       } else {
      //         out_vec[i] = 0;
      //       }
            
      //     }
      //     phi::Store(out_vec, quant_x + linear_index);
      // }
  }
}

template<typename T, int VecSize>
__global__ void QuantActKernel(const T* x, const int32_t rows, const int32_t cols, 
                                   const T* row_ranges, const int32_t* outlier_idx, int8_t* quant_x) {
  
  using InVec = phi::AlignedVector<T, VecSize>;
  using OutVec = phi::AlignedVector<int8_t, VecSize>;

  InVec in_vec;
  OutVec out_vec;
  for(int row_idx = blockIdx.x; row_idx < rows; row_idx += gridDim.x){
      for(int col_idx = threadIdx.x * VecSize; col_idx < cols; col_idx += blockDim.x * VecSize){
        int32_t linear_index = row_idx * cols + col_idx; 
        phi::Load<T, VecSize>(x + linear_index, &in_vec);
        #pragma unroll
        float scale = 1.0f / static_cast<float>(row_ranges[row_idx]);
        int32_t local_outlier_idx = outlier_idx[col_idx / 32];
        for (int i = 0; i < VecSize; ++i) {
          int32_t index = linear_index + i;
          if (local_outlier_idx & (1 << (index % 32))) {
            out_vec[i] = 0;
          } else {
            out_vec[i] = QuantFunc<T>()(in_vec[i], scale);
          }
        }
        phi::Store(out_vec, quant_x + linear_index);
      }
  }
}

template<typename T, int VecSize>
__global__ void Fill(T* input, T value, int64_t num) {
    phi::AlignedVector<T, VecSize> in_vec;
    int stride = blockDim.x * gridDim.x * VecSize;
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;

    for (int idx = base_idx; idx < num; idx += stride) {
        #pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            in_vec[j] = value;
        }
        phi::Store(in_vec, input + idx);
    }

}



template<typename T>
__global__ void SplitKernel(const T* x, const int8_t* weight, const T* weight_scale, const int32_t* outlier_idx, 
                       T* sub_x, T* sub_weight, 
                            int m, int k, int n, int num_outlier_idx, int kfp_num, 
                            int sub_x_elem_cnt, int sub_w_elem_cnt, int elem_cnt) {
  extern __shared__ int32_t k_ids_shm[]; 
  int32_t cnt = 0;
  
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int i = 0; i < kfp_num; ++i) {
      k_ids_shm[i] = -1;
    }
    
    for (int i = 0; i < num_outlier_idx; ++i) {
      int32_t outlier_id = outlier_idx[i];
      if (outlier_id == 0) continue;
      for (int j = 0; j < 32; ++j) {
        if (outlier_id & (1 << j)) {
          k_ids_shm[cnt++] = i * 32 + j;
        }
      }
    }
  }  
                    
  __syncthreads();
  
  for(int linear_idx=blockIdx.x * blockDim.x + threadIdx.x; linear_idx < elem_cnt; linear_idx+=blockDim.x * gridDim.x){
    int32_t row_idx = linear_idx / kfp_num; 
    int32_t col_idx = linear_idx % kfp_num;
    int32_t k_id = k_ids_shm[col_idx];
    if (k_id == -1) continue;
    if(linear_idx < sub_x_elem_cnt){
        sub_x[row_idx * kfp_num + col_idx] = x[row_idx * k + k_id];
    } 
    if(linear_idx < sub_w_elem_cnt){
        sub_weight[row_idx * kfp_num + col_idx] = DequantFunc<T>()(weight[row_idx * k + k_id], weight_scale[row_idx]);
    }
  }
}

__global__ void UpdateOutlier(int32_t* outlier_idx, int32_t* total_num){
    constexpr int IntSize = 32;

    int32_t outlier_val = outlier_idx[threadIdx.x]; 
    #pragma unroll 
    for(int i = 0; i < IntSize; i++){
      while (outlier_val) {
        outlier_val = outlier_val & (outlier_val-1);
        // ++kfp_num;
        atomicAdd(total_num, 1); 
      }
    }
}

// Input: x:int32:[m, n], x_fp16:T:[m, n], input_range:T:[m], weight_scale:T:[n]
// Outpuy: y:T:[m, n]

template <typename T, int VecSize>
__global__ void DequantMergeKernel(const int32_t* x, const T* x_fp, const T* input_range, const T* weight_scale, T* y, int m, int n) {
  using FpVec = phi::AlignedVector<T, VecSize>;
  using IntVec = phi::AlignedVector<int32_t, VecSize>;

  FpVec x_fp_vec;
  FpVec out_vec;
  IntVec x_vec;

  for(int row_idx = blockIdx.x; row_idx < m; row_idx += gridDim.x){
      for(int col_idx = threadIdx.x * VecSize; col_idx < n; col_idx += blockDim.x * VecSize){
        int linear_idx = row_idx * n + col_idx;
        phi::Load(x_fp + linear_idx, &x_fp_vec);
        phi::Load(x + linear_idx, &x_vec);
        #pragma unroll
        for (int i = 0; i < VecSize; ++i) {
          T dequant_x_fp = DequantFunc<T>()(x_vec[i], input_range[row_idx], weight_scale[col_idx + i]);
          out_vec[i] = x_fp_vec[i] + dequant_x_fp;
        }
        phi::Store(out_vec, y + linear_idx);
      }
  }
}

template<typename T>
void LaunchFillKernel(T* input, T value, int64_t num, GpuLaunchConfig* gpu_config, gpuStream_t stream) {
  constexpr int VecSize = 16 / sizeof(T);
  Fill<T, VecSize><<<gpu_config->block_per_grid, gpu_config->thread_per_block, 0, stream>>>(input, value, num);
}

template<typename T> 
void LaunchReduceAbsMaxQuantKernel(const T* x, const float threshold, const int32_t rows, const int32_t cols, 
                                   T* row_ranges, int32_t* outlier_idx, int8_t* quant_x, gpuStream_t stream) {
  constexpr int NumThreads=256;
  constexpr int VecSize= 16 / sizeof(T);

  using DataT = typename PDDataTypeTraits<T>::DataType; 

  ReduceAbsMaxKernel<DataT, VecSize><<<rows, NumThreads, 0, stream>>>(reinterpret_cast<const DataT*>(x), threshold, rows, cols, 
                                                                  reinterpret_cast<DataT*>(row_ranges), outlier_idx);
  QuantActKernel<DataT, VecSize><<<rows, NumThreads, 0, stream>>>(reinterpret_cast<const DataT*>(x), rows, cols, 
                                                                  reinterpret_cast<DataT*>(row_ranges), outlier_idx, quant_x);                                                            
}

// template<typename T>
// void LaunchSplitKernel(const T* x, const int8_t* weight, const T* weight_scale, const int32_t* outlier_idx, 
//                        T* sub_x, T* sub_weight, int m, int k, int n, int kfp_num, gpuStream_t stream) {
//   int NumThreads=kfp_num;
//   int max_row = m > n ? m : n;
//   int64_t num_outlier_idx = (k + 31) / 32;    


//   using DataT = typename PDDataTypeTraits<T>::DataType;
//   SplitKernel<DataT><<<max_row, NumThreads, kfp_num * sizeof(int32_t), stream>>>(reinterpret_cast<const DataT*>(x), weight, 
//                                                           reinterpret_cast<const DataT*>(weight_scale), outlier_idx, 
//                                                           reinterpret_cast<DataT*>(sub_x), reinterpret_cast<DataT*>(sub_weight), m, k, n, num_outlier_idx, kfp_num);            
// }

template<typename T>
void LaunchSplitKernel(const T* x, const int8_t* weight, const T* weight_scale, const int32_t* outlier_idx, 
                       T* sub_x, T* sub_weight, int m, int k, int n, int kfp_num, gpuStream_t stream) {
  // kfp_num 
  int NumThreads=256;
  constexpr int kNumWaves = 16; 
  int dev;
  cudaGetDevice(&dev);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  int tpm;
  cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
  int max_row = m > n ? m : n;
  const int elem_cnt = max_row * kfp_num; 
  const int grid = std::max<int>(1, std::min<int64_t>((elem_cnt + NumThreads - 1) / NumThreads,
                                                   sm_count * tpm / NumThreads * kNumWaves));
  int64_t num_outlier_idx = (k + 31) / 32;  

  const int32_t sub_x_elem_cnt = m * kfp_num; 
  const int32_t sub_w_elem_cnt = n * kfp_num; 

  using DataT = typename PDDataTypeTraits<T>::DataType;
  SplitKernel<DataT><<<grid, NumThreads, kfp_num * sizeof(int32_t), stream>>>(reinterpret_cast<const DataT*>(x), weight, 
                                                                  reinterpret_cast<const DataT*>(weight_scale), outlier_idx, 
                                                                     reinterpret_cast<DataT*>(sub_x), 
                                                                     reinterpret_cast<DataT*>(sub_weight), m, k, n, num_outlier_idx, 
                                                                     kfp_num, 
                                                                     sub_x_elem_cnt, 
                                                                     sub_w_elem_cnt, 
                                                                     elem_cnt);            
}

template <typename T>
void LaunchDequantMergeKernel(const int32_t* x, const T* x_fp, const T* input_range, const T* weight_scale, T* y, int m, int n, gpuStream_t stream) {
  constexpr int NumThreads=256;
  constexpr int VecSize=16 / sizeof(T);

  using DataT = typename PDDataTypeTraits<T>::DataType;     

  DequantMergeKernel<DataT, VecSize><<<m, NumThreads, 0, stream>>>(x, reinterpret_cast<const DataT*>(x_fp), 
                                                              reinterpret_cast<const DataT*>(input_range), 
                                                              reinterpret_cast<const DataT*>(weight_scale), 
                                                              reinterpret_cast<DataT*>(y), m, n);
}

template <typename T>
void LLMGemm(const phi::GPUContext& dev_ctx, 
             const phi::DenseTensor* weight,
             const phi::DenseTensor* input,
             const phi::DenseTensor* weight_scale,
             const float threshold,
             phi::DenseTensor* output,
             phi::DenseTensor* workspace,
             std::string name,
             int m, int k, int n) {
  // absmax, quant, outlier  
  int64_t num_outlier_idx = (k + 31) / 32;      
  phi::DenseTensor row_ranges, outlier_idx, quant_input;
  row_ranges.Resize({m});
  outlier_idx.Resize({num_outlier_idx});
  quant_input.Resize({m, k});
  dev_ctx.Alloc<T>(&row_ranges);
  dev_ctx.Alloc<int32_t>(&outlier_idx);
  dev_ctx.Alloc<int8_t>(&quant_input);

  PADDLE_ENFORCE_GPU_SUCCESS(	cudaMemsetAsync(outlier_idx.data<int32_t>(), 0, num_outlier_idx * sizeof(int32_t), dev_ctx.stream()));
  // VLOG(1) << "LaunchReduceAbsMaxQuantKernel";
  LaunchReduceAbsMaxQuantKernel(input->data<T>(), threshold, m, k, 
                          row_ranges.data<T>(), outlier_idx.data<int32_t>(), quant_input.data<int8_t>(), dev_ctx.stream());
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  // VLOG(1) << "input " << *input;
  VLOG(2) << "row_ranges " << row_ranges;
  VLOG(2) << "quant_input " << quant_input;
  // VLOG(1) << outlier_idx;

  // Test outlier
    // std::vector<int32_t> outlier_idx_vec(num_outlier_idx);
    // cudaMemcpy(outlier_idx_vec.data(), outlier_idx.data<int32_t>(), num_outlier_idx * sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t kfp_num = 0;
    phi::DenseTensor kfp_num_tensor;
    kfp_num_tensor.Resize({1});
    dev_ctx.Alloc<int32_t>(&kfp_num_tensor);


    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(kfp_num_tensor.data<int32_t>(), 0, sizeof(int32_t), dev_ctx.stream()));
    UpdateOutlier<<<1, num_outlier_idx, 0, dev_ctx.stream()>>>(outlier_idx.data<int32_t>(), kfp_num_tensor.data<int32_t>()); 
    cudaMemcpy(&kfp_num, kfp_num_tensor.data<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    VLOG(1) << "kfp_num = " << kfp_num;

  phi::DenseTensor sub_out;
  sub_out.Resize({m, n});       
  dev_ctx.Alloc<T>(&sub_out);       
  if (kfp_num != 0) {
    phi::DenseTensor sub_input, sub_weight;
    sub_input.Resize({m, kfp_num});
    sub_weight.Resize({n, kfp_num});
    
    dev_ctx.Alloc<T>(&sub_input);
    dev_ctx.Alloc<T>(&sub_weight);

    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(sub_input.data<T>(), 0, sub_input.numel() * sizeof(T), dev_ctx.stream()));
    


    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(sub_weight.data<T>(), 0, sub_weight.numel() * sizeof(T), dev_ctx.stream()));

    // VLOG(1) << "LaunchSplitKernel";

    LaunchSplitKernel(input->data<T>(), weight->data<int8_t>(), weight_scale->data<T>(), outlier_idx.data<int32_t>(), 
                        sub_input.data<T>(), sub_weight.data<T>(), m, k, n, kfp_num, dev_ctx.stream());


      CBLAS_TRANSPOSE transA = CblasNoTrans;
      CBLAS_TRANSPOSE transB = CblasTrans;
      T alpha = static_cast<T>(1.0);
      T beta = static_cast<T>(0.0);

      // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
      auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
      // VLOG(1) << "GEMM fp16 " << m << " " << n;
      blas.GEMM(transA,
                transB,
                m,
                n,
                kfp_num,
                alpha,
                sub_input.data<T>(),
                sub_weight.data<T>(),
                beta,
                sub_out.data<T>());

    // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(sub_out.data<T>(), 0, sub_out.numel() * sizeof(T), dev_ctx.stream()));
    
  }
  

  phi::DenseTensor int_out;
  int_out.Resize({m, n});
  dev_ctx.Alloc<int32_t>(&int_out);

  {

    auto helper = std::make_unique<CublasLtHelper<int32_t>>(m, k, n, dev_ctx.cublaslt_handle());
    // VLOG(1) << "GEMM int8";
    helper->GEMM(quant_input.data<int8_t>(),
                 weight->data<int8_t>(),
                 int_out.data<int32_t>(),
                 dev_ctx.stream(),
                 (void*)workspace->data<int8_t>(),
                  workspace->numel());
  }
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

  // VLOG(1) << "LaunchDequantMergeKernel";
  LaunchDequantMergeKernel<T>(int_out.data<int32_t>(), sub_out.data<T>(), row_ranges.data<T>(), weight_scale->data<T>(), output->data<T>(), m, n, dev_ctx.stream());
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
}


}  // namespace operators
}  // namespace paddle