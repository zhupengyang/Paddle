// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/default_fmha_grouped.h"
#include "paddle/phi/kernels/fusion/fused_multihead_attention_variable_kernel.h"
#include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/gemm/gemm_grouped.h"
#include "cutlass/util/device_memory.h"
#include "paddle/fluid/framework/tensor_util.h"


namespace phi {
namespace fusion {

using GemmCoord = cutlass::gemm::GemmCoord;

template <typename T>
struct DataTypeTraits {
  using DataType = T;
};

template <>
struct DataTypeTraits<cutlass::half_t> {
  using DataType = phi::dtype::float16;
};


struct Params {
  // meta params
  phi::DataType datatype;

  // [bs, nh, seq_len, dh]
  const void* query_ptr;
  const void* key_ptr;
  const void* value_ptr;

  // and it can be broadcasted in axis0, 1, 2.
  const void* mask_ptr = nullptr;

  const int* seq_lens = nullptr;

  // Output tensors
  void* output_ptr;        // [num_batches, num_heads, query_seq_len, head_size]
  void* output_accum_ptr = nullptr;  // [num_batches, num_heads, query_seq_len, head_size]

  // Scale
  float scale;

  // Dimensions/strides
  int32_t num_batches;
  int32_t num_heads;
  int32_t query_seq_len;
  int32_t key_value_seq_len;
  int32_t head_size;
  int32_t value_head_size;

  int64_t ldq;
  int64_t ldk;
  int64_t ldm;
  int64_t ldv;
  int64_t ldo;

  int64_t ElementQ;
  int64_t ElementK;
  int64_t ElementM;
  int64_t ElementV;
  int64_t ElementO;

  bool causal;
  bool mask_broadcast_row;
};

__global__ void get_problem_sizes(const int *seq_lens, 
                                  GemmCoord *problem_sizes0, 
                                  GemmCoord *problem_sizes1,
                                  const int bs,
                                  const int num_head,
                                  const int kv_seq_len,
                                  const int head_size,
                                  const int value_head_size) {
  int bi = blockIdx.x;
  int hi = threadIdx.x;
  if (bi < bs && hi < num_head) {
    int id = bi * num_head + hi;
    int m = seq_lens[bi];
    int mkv = kv_seq_len;
    int k0 = head_size;
    int k1 = value_head_size;
    GemmCoord problem0(m, mkv, k0);
    GemmCoord problem1(m, k1, mkv);
    problem_sizes0[id] = problem0;
    problem_sizes1[id] = problem1;
  }
}

template <typename T,
          typename ArchTag,
          bool IsAligned,
          bool MaskIsAligned,
          int QueriesPerBlock,
          int KeysPerBlock,
          bool SingleValueIteration,
          bool AddMask,
          bool MaskBroadcastRow>
void LaunchMultiHeadAttentionKernel(Params params,
                                    const phi::GPUContext& ctx) {
  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutP = cutlass::layout::RowMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;
  
  using AttentionKernel = typename cutlass::gemm::kernel::DefaultFMHAGrouped<
      T,      
      ArchTag, 
      IsAligned,
      MaskIsAligned,
      QueriesPerBlock,
      KeysPerBlock,
      SingleValueIteration,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      AddMask,
      MaskBroadcastRow>::FMHAKernel; // kHostPrecompute kDeviceOnly
  using FMHA = cutlass::gemm::device::GemmGrouped<AttentionKernel>;
  using scalar_t = typename FMHA::GemmKernel::scalar_t;
  using accum_t = typename FMHA::GemmKernel::accum_t;
  using output_t = typename FMHA::GemmKernel::output_t;
  using output_accum_t = typename FMHA::GemmKernel::output_accum_t;
  using ElementQ = scalar_t;
  using ElementK = scalar_t;
  using ElementP = accum_t;
  using ElementM = scalar_t;
  using ElementAccumulator = accum_t;
  using ElementV = scalar_t;
  using ElementO = output_t;
  using ElementOAccum = output_accum_t;
  using data_t = typename DataTypeTraits<T>::DataType;

  int problem_count = params.num_batches * params.num_heads;
  
  std::vector<GemmCoord> problem_sizes1;
  problem_sizes1.reserve(problem_count);

  paddle::memory::AllocationPtr problem_sizes_device0{nullptr};
  paddle::memory::AllocationPtr problem_sizes_device1{nullptr};
  problem_sizes_device0 = paddle::memory::Alloc(
          ctx.GetPlace(),
          problem_count * sizeof(GemmCoord),
          phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  problem_sizes_device1 = paddle::memory::Alloc(
          ctx.GetPlace(),
          problem_count * sizeof(GemmCoord),
          phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  GemmCoord *problem0_device = reinterpret_cast<GemmCoord*>(problem_sizes_device0->ptr());
  GemmCoord *problem1_device = reinterpret_cast<GemmCoord*>(problem_sizes_device1->ptr());
  get_problem_sizes<<<params.num_batches, params.num_heads, 0, ctx.stream()>>>(params.seq_lens, 
                                                                               problem0_device, 
                                                                               problem1_device, 
                                                                               params.num_batches, 
                                                                               params.num_heads, 
                                                                               params.key_value_seq_len, 
                                                                               params.head_size, 
                                                                               params.value_head_size);
  paddle::memory::Copy(phi::CPUPlace(), problem_sizes1.data(), ctx.GetPlace(), problem1_device, sizeof(GemmCoord) * problem_count, ctx.stream());

  if (AttentionKernel::kNeedsOutputAccumulatorBuffer) {
    const int64_t output_size = params.num_batches * params.num_heads *
                                params.query_seq_len * params.value_head_size;
    paddle::memory::AllocationPtr tmp_output_accum_buffer_ptr{nullptr};
    tmp_output_accum_buffer_ptr = paddle::memory::Alloc(
        ctx.GetPlace(),
        output_size * sizeof(ElementOAccum),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    params.output_accum_ptr = tmp_output_accum_buffer_ptr->ptr();
  } 

  int threadblock_count = FMHA::sufficient(problem_sizes1.data(), problem_count);
  typename FMHA::Arguments args(
    problem0_device,
    problem1_device,
    problem_count,
    threadblock_count,
    params.num_heads,
    const_cast<T*>(reinterpret_cast<const T*>(params.query_ptr)),
    const_cast<T*>(reinterpret_cast<const T*>(params.key_ptr)),
    params.mask_ptr ? const_cast<T*>(reinterpret_cast<const T*>(params.mask_ptr)) : nullptr,
    const_cast<T*>(reinterpret_cast<const T*>(params.value_ptr)),
    reinterpret_cast<T*>(params.output_ptr),
    AttentionKernel::kNeedsOutputAccumulatorBuffer ? reinterpret_cast<output_accum_t*>(params.output_accum_ptr) : nullptr,
    params.ldq,
    params.ldk,
    params.ldm,
    params.ldv,
    params.ldo,
    params.ElementQ,
    params.ElementK,
    params.ElementM,
    params.ElementV,
    params.ElementO,
    params.causal,
    params.scale,
    problem_sizes1.data()
  );

  FMHA fmha;
  cutlass::Status status;
  size_t workspace_size = fmha.get_workspace_size(args);
  phi::DenseTensor workspace;
  workspace.Resize(phi::make_ddim({static_cast<int64_t>(workspace_size)}));
  ctx.template Alloc<uint8_t>(&workspace);
  status = fmha.initialize(args, workspace.data<uint8_t>());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize CUTLASS Grouped FMHA kernel." << std::endl;
  }
  status = fmha.run();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to run CUTLASS Grouped FMHA kernel." << std::endl;
  }
}

template <typename T,
          typename ArchTag,
          bool IsAligned,
          bool MaskIsAligned,
          int QueriesPerBlock,
          int KeysPerBlock,
          bool SingleValueIteration,
          bool AddMask>
void DispatchFMHAMaskBroadcastRow(Params params,
                                  const phi::GPUContext& ctx) {
  if (params.mask_broadcast_row) {
    LaunchMultiHeadAttentionKernel<T,
                                   ArchTag,
                                   IsAligned,
                                   MaskIsAligned,
                                   QueriesPerBlock,
                                   KeysPerBlock,
                                   SingleValueIteration,
                                   AddMask,
                                   true>(params, ctx);
  } else {
    LaunchMultiHeadAttentionKernel<T,
                                   ArchTag,
                                   IsAligned,
                                   MaskIsAligned,
                                   QueriesPerBlock,
                                   KeysPerBlock,
                                   SingleValueIteration,
                                   AddMask,
                                   false>(params, ctx);
  }
}

template <typename T,
          typename ArchTag,
          bool IsAligned,
          bool MaskIsAligned,
          int QueriesPerBlock,
          int KeysPerBlock,
          bool SingleValueIteration>
void DispatchFMHAAddMask(Params params, const phi::GPUContext& ctx) {
  if (params.mask_ptr != nullptr) {
    DispatchFMHAMaskBroadcastRow<T,
                                 ArchTag,
                                 IsAligned,
                                 MaskIsAligned,
                                 QueriesPerBlock,
                                 KeysPerBlock,
                                 SingleValueIteration,
                                 true>(params, ctx);
  } else {
    DispatchFMHAMaskBroadcastRow<T,
                                 ArchTag,
                                 IsAligned,
                                 MaskIsAligned,
                                 QueriesPerBlock,
                                 KeysPerBlock,
                                 SingleValueIteration,
                                 false>(params, ctx);
  }
}

template <typename T,
          typename ArchTag,
          bool IsAligned,
          bool MaskIsAligned,
          int QueriesPerBlock,
          int KeysPerBlock>
void DispatchFMHASingleValueIteration(Params params,
                                      const phi::GPUContext& ctx) {
  if (params.value_head_size <= KeysPerBlock) {
    DispatchFMHAAddMask<T,
                        ArchTag,
                        IsAligned,
                        MaskIsAligned,
                        QueriesPerBlock,
                        KeysPerBlock,
                        true>(params, ctx);
  } else {
    DispatchFMHAAddMask<T,
                        ArchTag,
                        IsAligned,
                        MaskIsAligned,
                        QueriesPerBlock,
                        KeysPerBlock,
                        false>(params, ctx);
  }
}

template <typename T, typename ArchTag, bool IsAligned, bool MaskIsAligned>
void DispatchFMHABlockSize(Params params, const phi::GPUContext& ctx) {
  if (params.value_head_size > 64) {
    DispatchFMHASingleValueIteration<T, ArchTag, IsAligned, MaskIsAligned, 32, 128>(params,
                                                                     ctx);
  } else {
    DispatchFMHASingleValueIteration<T, ArchTag, IsAligned, MaskIsAligned, 64, 64>(params,
                                                                    ctx);
  }
}

template <typename T, typename ArchTag, bool IsAligned>
void DispatchFMHAMaskIsAligned(Params params, const phi::GPUContext& ctx) {
  if (reinterpret_cast<uintptr_t>(params.mask_ptr) % 16 == 0 &&
      params.key_value_seq_len % (16 / sizeof(T)) == 0) {
    DispatchFMHABlockSize<T, ArchTag, IsAligned, true>(params, ctx);
  } else {
    DispatchFMHABlockSize<T, ArchTag, IsAligned, false>(params, ctx);
  }
}

template <typename T, typename ArchTag>
void DispatchFMHAIsAligned(Params params, const phi::GPUContext& ctx) {
  if (reinterpret_cast<uintptr_t>(params.query_ptr) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(params.key_ptr) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(params.value_ptr) % 16 == 0 && 
      params.head_size % (16 / sizeof(T)) == 0 && params.value_head_size % (16 / sizeof(T)) == 0) {
    DispatchFMHAMaskIsAligned<T, ArchTag, true>(params, ctx);
  } else {
    DispatchFMHAMaskIsAligned<T, ArchTag, false>(params, ctx);
  }
}

template <typename T>
void DispatchFMHAArchTag(Params params, const phi::GPUContext& ctx) {
  const int compute_capability = ctx.GetComputeCapability();
  // if (compute_capability == 80) {
  //   DispatchFMHAIsAligned<T, cutlass::arch::Sm80>(params, ctx);
  // }  else {
  //   return;
  // }

  // LaunchMultiHeadAttentionKernel<T, cutlass::arch::Sm80, true, false, 32, 128, false, true, false>(params, ctx);

  if (compute_capability == 80) {
    DispatchFMHAIsAligned<T, cutlass::arch::Sm80>(params, ctx);
  } else if (compute_capability == 75) {
    DispatchFMHAIsAligned<T, cutlass::arch::Sm75>(params, ctx);
  } else if (compute_capability == 70) {
    DispatchFMHAIsAligned<T, cutlass::arch::Sm70>(params, ctx);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Currently cutlass fused multihead attention kernel "
        "only support arch: SM80, SM75, SM70"));
    return;
  }
}

void DispatchFusedMultiheadAttentionKernel(Params params,
                                           const phi::GPUContext& ctx) {
  if (params.datatype == DataType::FLOAT32) {
    return DispatchFMHAArchTag<float>(params, ctx);
  } else if (params.datatype == DataType::FLOAT16) {
    return DispatchFMHAArchTag<cutlass::half_t>(params, ctx);
  } else {
    PADDLE_ENFORCE_EQ(true,
                      false,
                      phi::errors::Unimplemented(
                          "Currently cutlass fused multihead attention kernel "
                          "only support datatype: float32 and float16. "));
    return;
  }
}

template <typename T, typename Context>
void MultiHeadAttentionVariableForwardKernel(const Context& ctx,
                                             const DenseTensor& query,
                                             const DenseTensor& key,
                                             const DenseTensor& value,
                                             const DenseTensor& seq_lens,
                                             const paddle::optional<DenseTensor>& mask,
                                             const float scale,
                                             const bool causal,
                                             DenseTensor* output) {
  ctx.template Alloc<T>(output);

  Params params{};
  // [B, N, S, H]
  params.seq_lens = seq_lens.data<int>();

  params.num_batches = query.dims()[0];
  params.num_heads = query.dims()[1];
  params.query_seq_len = query.dims()[2];
  params.head_size = query.dims()[3];
  params.key_value_seq_len = key.dims()[2];
  params.value_head_size = value.dims()[3];

  params.datatype = query.dtype();
  params.query_ptr = query.data();
  params.key_ptr = key.data();
  params.value_ptr = value.data();

  params.output_ptr = output->data();

  params.ldq = params.head_size;
  params.ldk = params.head_size;
  params.ldv = params.value_head_size;
  params.ldo = params.value_head_size;

  params.ElementQ = params.query_seq_len * params.head_size;
  params.ElementK = params.key_value_seq_len * params.head_size;
  params.ElementV = params.key_value_seq_len * params.value_head_size;
  params.ElementO = params.query_seq_len * params.value_head_size;

  params.scale = scale;
  params.causal = causal;

  if (mask) {
    // [B, 1, S, D]
    params.ldm = params.key_value_seq_len;
    params.ElementM = params.query_seq_len * params.key_value_seq_len;
    auto mask_tensor = mask.get();
    params.mask_ptr = mask_tensor.data();
    params.mask_broadcast_row = false;
  }

  DispatchFusedMultiheadAttentionKernel(params, ctx);
}

template <typename T, typename Context>
void MultiHeadAttentionVariableWrapper(const Context& ctx,
                                       T* query,
                                       T* key,
                                       T* value,
                                       const int* seq_lens,
                                       const T* mask,
                                       float scale,
                                       bool causal,
                                       int64_t batch_size, 
                                       int64_t num_heads, 
                                       int64_t seq_len, 
                                       int64_t out_seq_len, 
                                       int64_t head_size,
                                       int64_t value_head_size,
                                       T* output) {
  Params params{};
  // [B, N, S, H]
  params.seq_lens = seq_lens;

  params.num_batches = batch_size;
  params.num_heads = num_heads;
  params.query_seq_len = seq_len;
  params.head_size = head_size;
  params.key_value_seq_len = out_seq_len;
  params.value_head_size = value_head_size;

  params.datatype = DataType::FLOAT16;
  params.query_ptr = query;
  params.key_ptr = key;
  params.value_ptr = value;

  params.output_ptr = output;

  params.ldq = params.head_size;
  params.ldk = params.head_size;
  params.ldv = params.value_head_size;
  params.ldo = params.value_head_size;

  params.ElementQ = params.query_seq_len * params.head_size;
  params.ElementK = params.key_value_seq_len * params.head_size;
  params.ElementV = params.key_value_seq_len * params.value_head_size;
  params.ElementO = params.query_seq_len * params.value_head_size;

  params.scale = scale;
  params.causal = causal;

  if (mask) {
    // [B, 1, S, D]
    params.ldm = params.key_value_seq_len;
    params.ElementM = params.query_seq_len * params.key_value_seq_len;
    params.mask_ptr = mask;
    params.mask_broadcast_row = false;
  }
  DispatchFusedMultiheadAttentionKernel(params, ctx);
}

template void MultiHeadAttentionVariableWrapper(
    const phi::GPUContext& ctx,
    phi::dtype::float16* query,
    phi::dtype::float16* key,
    phi::dtype::float16* value,
    const int* seq_lens,
    const phi::dtype::float16* mask,
    const float scale,
    const bool causal,
    const int64_t batch_size, 
    const int64_t num_heads, 
    const int64_t seq_len, 
    const int64_t out_seq_len, 
    const int64_t head_size,
    const int64_t value_head_size,
    phi::dtype::float16* output
);

template void MultiHeadAttentionVariableWrapper(
    const phi::GPUContext& ctx,
    float* query,
    float* key,
    float* value,
    const int* seq_lens,
    const float* mask,
    const float scale,
    const bool causal,
    const int64_t batch_size, 
    const int64_t num_heads, 
    const int64_t seq_len, 
    const int64_t out_seq_len, 
    const int64_t head_size,
    const int64_t value_head_size,
    float* output
);

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    fused_multihead_attention_variable,
    GPU,
    ALL_LAYOUT,
    phi::fusion::MultiHeadAttentionVariableForwardKernel,
    float,
    phi::dtype::float16) { kernel->InputAt(3).SetDataType(phi::DataType::INT32); }
