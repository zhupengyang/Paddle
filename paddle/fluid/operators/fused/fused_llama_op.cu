/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/custom_all_reduce.h"
#include "paddle/fluid/operators/fused/fused_multi_transformer_op.cu.h"
#include "paddle/fluid/operators/custom_all_reduce.h"

#include "paddle/fluid/platform/device/gpu/gpu_resource_pool.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/rms_norm_kernel.h"

#include<algorithm>


DECLARE_bool(use_cutlass_fmha); 
DECLARE_int64(custom_allreduce_one_shot_threshold);
DECLARE_int64(custom_allreduce_two_shot_threshold);

namespace paddle {
namespace operators {

static CustomNCCLComm *GetCustomNCCLComm(const phi::GPUContext &ctx,
                                         int ring_id) {
  static auto comm =
      CreateCustomNCCLComm(ctx,
                           FLAGS_custom_allreduce_one_shot_threshold,
                           FLAGS_custom_allreduce_two_shot_threshold,
                           ring_id);
  return comm.get();
}

static phi::DenseTensor CustomAllReduce(const phi::DenseTensor &t) {
  auto *ctx = static_cast<phi::GPUContext *>(
      platform::DeviceContextPool::Instance().Get(t.place()));
  auto comm = GetCustomNCCLComm(*ctx, 0);
  PADDLE_ENFORCE_NOT_NULL(comm);
  phi::DenseTensor ret;
  ret.Resize(t.dims());
  ctx->Alloc(&ret, t.dtype());
  comm->SwapInput(&ret);
  phi::Copy(*ctx, t, t.place(), false, &ret);
  return comm->AllReduce();
}

template <typename T>
class FusedLLAMAOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;

    auto &dev_ctx = ctx.cuda_device_context();

    auto *time_step = ctx.Input<phi::DenseTensor>("TimeStep");
    // 0. input
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    const auto input_x_dims = input_x->dims();
    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int bsz_seq = bsz * seq_len;
    const std::string act_method = ctx.Attr<std::string>("act_method");

    // Note(Zhengzekang): LLAMA use pre layernorm architecture. 
    // bool pre_layer_norm = true; 

    bool remove_padding = false;
    auto *sequence_lengths = ctx.Input<phi::DenseTensor>("SeqLengths");
    if (sequence_lengths) {
      remove_padding = true;
    }
    phi::DenseTensor d_token_tensor;
    phi::DenseTensor padding_offset_tensor;
    phi::DenseTensor x_remove_padding;
    bool encoder_remove_padding = (remove_padding && !time_step);
    int token_num = 0;

    // remove padding in encoder
    if (encoder_remove_padding) {
      // just for encoder
      d_token_tensor.Resize({{1}});
      auto *d_token_num = dev_ctx.Alloc<int>(
          &d_token_tensor, d_token_tensor.numel() * sizeof(int));
      // alloc the max size of padding_offset_tensor
      padding_offset_tensor.Resize({{bsz_seq}});
      dev_ctx.Alloc<int>(&padding_offset_tensor,
                         padding_offset_tensor.numel() * sizeof(int));
      InvokeGetPaddingOffset(dev_ctx,
                             &token_num,
                             d_token_num,
                             padding_offset_tensor.data<int>(),
                             sequence_lengths->data<int>(),
                             bsz,
                             seq_len);
      padding_offset_tensor.Resize({{token_num}});
      x_remove_padding.Resize({{token_num, dim_embed}});
      dev_ctx.Alloc<T>(&x_remove_padding, x_remove_padding.numel() * sizeof(T));
      InvokeRemovePadding(dev_ctx,
                          x_remove_padding.data<T>(),
                          input_x->data<T>(),
                          padding_offset_tensor.data<int>(),
                          token_num,
                          dim_embed);
    } else {
      token_num = bsz_seq;
    }
    printf("Token num is: %d \n", token_num); 

    if (token_num == 0) {
      return;
    }

    auto *padding_offset_data =
        encoder_remove_padding ? padding_offset_tensor.data<int>() : nullptr;

    // 1. RMSNorm
    /*
    Note(zhengzekang): LLAMA RMSNorm weight type is as same as Input, RMSNorm do not need bias. 
    Since this OP is only used in Inference, we do not save variance for backward. 
    */ 
    const float epsilon = ctx.Attr<float>("epsilon");
    auto ln_scales = ctx.MultiInput<phi::DenseTensor>("LnScale");


    // 2. qkv
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto qkv_weights = ctx.MultiInput<phi::DenseTensor>("QKVW");
    const bool trans_qkvw = ctx.Attr<bool>("trans_qkvw");
    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    // Note(zhengzekang): LLAMA Matmul do not need bias. 
    bool qkv_compute_bias = false;
    // (transA, transB, qkv_compute_bias) = (false, trans_qkvw, false)
    // Since we fused QKVBias into QKVBiasAddTransposeSplit kernel, here we
    // set qkv_compute_bias as false.
    auto qkv_compute = AttnMatMul<T>(dev_ctx,
                                     false,
                                     trans_qkvw,
                                     token_num,
                                     output_size,
                                     input_size,
                                     /*qkv_compute_bias=*/qkv_compute_bias);
    phi::DenseTensor qkv_out;
    qkv_out.Resize({{token_num, 3, num_head, dim_head}});
    auto *qkv_out_data =
        dev_ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

    // 2.1 rotary
    auto *rotary_tensor = ctx.Input<phi::DenseTensor>("RotaryPosEmb");
    const int rotary_emb_dims = ctx.Attr<int>("rotary_emb_dims");

    // 3. fmha
    AttnDropoutParam attn_param(
        true, "upscale_in_train", 0.0, true, true, 0, nullptr);
    auto fmha_compute =
        FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
    auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");
    auto cache_kvs = ctx.MultiInput<phi::DenseTensor>("CacheKV");
    auto cache_kv_outs = ctx.MultiOutput<phi::DenseTensor>("CacheKVOut");
    auto pre_caches = ctx.MultiInput<phi::DenseTensor>("PreCaches");
    int cache_offset = 0;
    if (pre_caches.size() > 0) {
      cache_offset = pre_caches[0]->dims()[3];
    }

    auto out_seq_len = seq_len;
    if (time_step) {
      PADDLE_ENFORCE_EQ(time_step->place(),
                        platform::CPUPlace(),
                        platform::errors::PreconditionNotMet(
                            "The place of input(TimeStep) must be CPUPlace."));
      // cache_seq_len
      int time_step_value = time_step->data<int>()[0];
      PADDLE_ENFORCE_GT(time_step_value,
                        0,
                        platform::errors::PreconditionNotMet(
                            "The value of time_step must > 0, but now is %d",
                            time_step_value));
      PADDLE_ENFORCE_EQ(
          seq_len,
          1,
          platform::errors::PreconditionNotMet(
              "In decode stage, the seq_len of input must be 1, but now is %d",
              seq_len));
      out_seq_len += time_step_value;
    } else {
      out_seq_len += cache_offset;
    }

    phi::DenseTensor q_transpose_out, kv_transpose_out, qk_out;
    q_transpose_out.Resize({{bsz, num_head, seq_len, dim_head}});
    auto *q_transpose_out_data =
        dev_ctx.Alloc<T>(&q_transpose_out, q_transpose_out.numel() * sizeof(T));

    kv_transpose_out.Resize({{2, bsz, num_head, seq_len, dim_head}});
    auto *kv_transpose_out_data = dev_ctx.Alloc<T>(
        &kv_transpose_out, kv_transpose_out.numel() * sizeof(T));

    if (encoder_remove_padding) {
      InitValue(dev_ctx, q_transpose_out_data, q_transpose_out.numel(), static_cast<T>(0.));
      InitValue(dev_ctx, kv_transpose_out_data, kv_transpose_out.numel(), static_cast<T>(0.));
    }

    if (!FLAGS_use_cutlass_fmha) {
      qk_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
      auto *qk_out_data = dev_ctx.Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));
    }

    phi::DenseTensor src_mask_out;
    if (!FLAGS_use_cutlass_fmha) {
      if (cache_offset > 0) {
        src_mask_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
        auto *src_mask_out_data =
            dev_ctx.Alloc<T>(&src_mask_out, src_mask_out.numel() * sizeof(T));
      }
    }

    // [2, bs, num_head, cache_seq_len + seq_len, head_dim]
    phi::DenseTensor pre_cache_kv_out;
    if (cache_offset > 0) {
      pre_cache_kv_out.Resize(
          {{2, bsz, num_head, seq_len + cache_offset, dim_head}});
      auto *pre_cache_kv_out_data = dev_ctx.Alloc<T>(
          &pre_cache_kv_out, pre_cache_kv_out.numel() * sizeof(T));
    }

    phi::DenseTensor softmax_out;
    phi::DenseTensor attn_dropout_mask_out, attn_dropout_out;
    phi::DenseTensor qktv_out, fmha_out;
    if (!FLAGS_use_cutlass_fmha) {
      softmax_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
      auto *softmax_out_data =
          dev_ctx.Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));
    }

    T *attn_dropout_mask_out_data = nullptr;
    T *attn_dropout_data_data = nullptr;

    qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
    auto *qktv_out_data =
        dev_ctx.Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
    fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
    auto *fmha_out_data =
        dev_ctx.Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<phi::DenseTensor>("OutLinearW");
    int ring_id = ctx.Attr<int>("ring_id");
    auto *custom_comm = GetCustomNCCLComm(dev_ctx, ring_id);
    // (transA, transB, qkv_compute_bias) = (false, false, false)
    auto out_linear_compute = AttnMatMul<T>(
        dev_ctx, false, false, token_num, dim_embed, hidden_size, false);

    // 5. ln(residual)
    auto ffn_ln_scales = ctx.MultiInput<phi::DenseTensor>("FFNLnScale");
    phi::DenseTensor residual_out, dropout_mask_out;
    T *residual_out_data = nullptr;

    residual_out.Resize({{token_num, dim_embed}});
    residual_out_data =
        dev_ctx.Alloc<T>(&residual_out,
                        residual_out.numel() * sizeof(T));

    uint8_t *dropout_mask_out_data = nullptr;

    // 6. ffn matmul1. LLAMA use swiGLU. Author(zhengzekang)
    auto ffn1_weights = ctx.MultiInput<phi::DenseTensor>("FFN1Weight");
    auto ffn1_weight_dim = ffn1_weights[0]->dims();
    int dim_ffn = ffn1_weight_dim[1];
    FFNGluHelper<T> ffn1_glu_helper(
        dev_ctx, "swiglu", token_num, dim_ffn / 2, dim_ffn, dim_embed);

    phi::DenseTensor ffn1_out;
    ffn1_out.Resize({{token_num, dim_ffn}});
    auto *ffn1_out_data =
        dev_ctx.Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

    int tmp_dim_ffn = dim_ffn / 2;

    int8_t *ffn1_dropout_mask_data = nullptr;
    phi::DenseTensor ffn1_dropout_out, ffn1_dropout_mask;
    ffn1_dropout_out.Resize({{token_num, tmp_dim_ffn}});
    auto *ffn1_dropout_out_data = dev_ctx.Alloc<T>(
        &ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));

    // 8. ffn2 matmul
    auto ffn2_weights = ctx.MultiInput<phi::DenseTensor>("FFN2Weight");
    auto ffn2_linear_compute = AttnMatMul<T>(
        dev_ctx, false, false, token_num, dim_embed, tmp_dim_ffn, /*compute_bias*/false);

    // 9. ffn2 residual
    DropoutParam ffn2_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> ffn2_fused_dropout_helper(
        dev_ctx, token_num, dim_embed, ffn2_dropout_param, epsilon);

    // calc
    auto *out = ctx.Output<phi::DenseTensor>("Out");
    auto *from_data = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));

    // Init out
    if (encoder_remove_padding) {
      InitValue(dev_ctx, from_data, out->numel(), static_cast<T>(0.));
    }

    phi::DenseTensor tmp_out, tmp_out_rm_padding;
    tmp_out.Resize({{token_num, dim_embed}});
    if (encoder_remove_padding) {
      tmp_out_rm_padding.Resize({{token_num, dim_embed}});
      auto *tmp_out_rm_padding_data = dev_ctx.Alloc<T>(
          &tmp_out_rm_padding, tmp_out_rm_padding.numel() * sizeof(T));
    }
    auto *tmp_out_data =
        dev_ctx.Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

    const T *x_data;
    if (encoder_remove_padding) {
      x_data = x_remove_padding.data<T>();
    } else {
      x_data = input_x->data<T>();
    }
    phi::DenseTensor *buf0 = nullptr;
    phi::DenseTensor *buf1 = nullptr;

    // step0:  x   --> buf1
    // step1: buf1 --> buf0
    // step2: buf0 --> buf1
    int layers = qkv_weights.size();
    if (encoder_remove_padding) {
      // In the case of variable lengths, the padding needs to be rebuilt
      // eventually. So buf0 and buf1 do not need to be changed according to the
      // pre_layer_norm and the number of layers.
      buf0 = &tmp_out;
      buf1 = &tmp_out_rm_padding;
    } else {
      if (layers & 1) {
        // odd, set buf1 as out
        buf0 = &tmp_out;
        buf1 = out;
      } else {
        // even, set buf0 as out
        buf0 = out;
        buf1 = &tmp_out;
      }
    }

    for (int i = 0; i < layers; ++i) {
      // step1. layer_norm, LLAMA use pre_layer_norm. 
      if (i == 0) {
        VLOG(2) << "step rmsnorm";
        // Here use rmsnorm, LLAMA RMSNorm weight dtype is same as input, and it do not need bias. Author(zhengzekang). 
        auto *ln_scale_data = ln_scales[i]->data<T>();
        phi::RmsNormWrapper<T, phi::GPUContext>(
          dev_ctx, 
          x_data,
          ln_scale_data, 
          epsilon, 
          token_num, 
          dim_embed,
          buf1->data<T>());
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step1";
      VLOG(0) << "rmsnorm 1_out:" << *buf1;
#endif

      // step2. qkv
      const phi::DenseTensor *qkv_bias = nullptr;
      VLOG(5)<< "Doing qkv gemm, mnk:" << token_num << ", " << output_size << ", " << input_size;
      qkv_compute.ComputeForward(
          qkv_weights[i], buf1, /*bias*/nullptr, &qkv_out, &qkv_out, true);

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step2";
      VLOG(0) << "qkv_out:" << qkv_out;
#endif

      // step3. fmha
      const phi::DenseTensor *cache_kv =
          cache_kvs.size() > 0 ? cache_kvs[i] : nullptr;
      phi::DenseTensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;

      if (time_step) {  // generation decoder stage
        printf("====== generation decode ===== \n"); 

        // [2, batch_size, num_head, max_seq_len, head_size]
        int max_seq_len = cache_kv->dims()[3];
        fmha<T>(dev_ctx,
                qkv_out,
                *qkv_bias, /*nullptr, Because LLAMA donot need bias*/
                *src_mask,
                sequence_lengths,
                rotary_tensor,
                cache_kv_out,
                &fmha_out,
                bsz,
                max_seq_len,
                num_head,
                dim_head,
                src_mask->dims()[3] - 1,
                rotary_emb_dims,
                1. / sqrt(dim_head), 
                false);
      } else if (cache_kv_out) {  // generation context stage
        printf("====== generation context ===== \n"); 

        const phi::DenseTensor *pre_cache_kv_tensor =
            pre_caches.size() > 0 ? pre_caches[i] : nullptr;
        phi::DenseTensor *pre_cache_kv_out_tmp =
            cache_offset > 0 ? &pre_cache_kv_out : nullptr;
        phi::DenseTensor *src_mask_tmp =
            cache_offset > 0 ? &src_mask_out : nullptr;
        const int *sequence_lengths_data =
              encoder_remove_padding ? sequence_lengths->data<int>() : nullptr;
        qkv_bias_add_transpose_split<T>(dev_ctx,
                                        q_transpose_out_data,
                                        kv_transpose_out_data,
                                        qkv_out_data,
                                        /*qkv_bias*/nullptr,
                                        padding_offset_data,
                                        token_num,
                                        bsz,
                                        num_head,
                                        seq_len,
                                        dim_head,
                                        qkv_compute_bias);

        // q_transpose_out_data [bs, head_num, seq_len, dim_head]
        // kv_transpose_out_data [2， bs, head_num, seq_len, dim_head]
        if (rotary_emb_dims != 0) {
          auto *rotary_emb_data = rotary_tensor->data<T>();
          const int *sequence_lengths_data =
              encoder_remove_padding ? sequence_lengths->data<int>() : nullptr;
          rotary_qk(dev_ctx,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    rotary_emb_data,
                    sequence_lengths_data,
                    rotary_emb_dims,
                    bsz,
                    num_head,
                    seq_len,
                    dim_head);
        }

        phi::DenseTensor *tmp_padding_offset_tensor =
            encoder_remove_padding ? &padding_offset_tensor : nullptr;
        if(FLAGS_use_cutlass_fmha){
          fmha_compute.ComputeForwardWithCutlassFMHA(pre_cache_kv_tensor,
                                                     src_mask,
                                                     tmp_padding_offset_tensor,
                                                     sequence_lengths,
                                                     &q_transpose_out,
                                                     &kv_transpose_out,
                                                     pre_cache_kv_out_tmp,
                                                     &qk_out,
                                                     src_mask_tmp,
                                                     &softmax_out,
                                                     &attn_dropout_mask_out,
                                                     &attn_dropout_out,
                                                     &qktv_out,
                                                     &fmha_out,
                                                     token_num);
        } else {
          fmha_compute.ComputeForwardWithoutTranspose(pre_cache_kv_tensor,
                                                      src_mask,
                                                      tmp_padding_offset_tensor,
                                                      &q_transpose_out,
                                                      &kv_transpose_out,
                                                      pre_cache_kv_out_tmp,
                                                      &qk_out,
                                                      src_mask_tmp,
                                                      &softmax_out,
                                                      &attn_dropout_mask_out,
                                                      &attn_dropout_out,
                                                      &qktv_out,
                                                      &fmha_out,
                                                      token_num);
        }
        
        const T *k_ptr = nullptr;
        const T *v_ptr = nullptr;

        if (cache_offset > 0) {
          // [2, bsz, num_head, cache_offset + seq_len, head_dim]
          const T *kv_data = pre_cache_kv_out.data<T>();
          k_ptr = kv_data;
          int64_t k_size = bsz * num_head * (seq_len + cache_offset) * dim_head;
          v_ptr = k_ptr + k_size;
        } else {
          // [3, bsz, num_head, seq_len, head_dim]
          int64_t k_size = bsz * seq_len * num_head * dim_head;
          const T *q_ptr = q_transpose_out_data;
          k_ptr = kv_transpose_out_data;
          v_ptr = k_ptr + k_size;
        }

        // [2, bsz, num_head, max_seq_len, head_dim]
        int max_seq_len = cache_kv_out->dims()[3];
        T *cache_kv_data = cache_kv_out->data<T>();
        int64_t cache_k_size = bsz * num_head * max_seq_len * dim_head;

        T *cache_k_ptr = cache_kv_data;
        T *cache_v_ptr = cache_kv_data + cache_k_size;

        const int seq_len_tmp = seq_len + cache_offset;
        write_cache_kv<T>(dev_ctx,
                          cache_k_ptr,
                          cache_v_ptr,
                          k_ptr,
                          v_ptr,
                          sequence_lengths_data,
                          bsz,
                          num_head,
                          seq_len_tmp,
                          max_seq_len,
                          dim_head);
      } else {  // not generation
        printf("====== not generation ===== \n"); 
        // TODO(wangxi): can remove dropout in inference
        qkv_bias_add_transpose_split<T>(dev_ctx,
                                        q_transpose_out_data,
                                        kv_transpose_out_data,
                                        qkv_out_data,
                                        /*qkv_bias*/nullptr,
                                        padding_offset_data,
                                        token_num,
                                        bsz,
                                        num_head,
                                        seq_len,
                                        dim_head,
                                        qkv_compute_bias);

        // q_transpose_out_data [bs, head_num, seq_len, dim_head]
        // kv_transpose_out_data [2， bs, head_num, seq_len, dim_head]
        if (rotary_emb_dims != 0) {
          auto *rotary_emb_data = rotary_tensor->data<T>();
          const int *sequence_lengths_data =
              encoder_remove_padding ? sequence_lengths->data<int>() : nullptr;
          rotary_qk(dev_ctx,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    rotary_emb_data,
                    sequence_lengths_data,
                    rotary_emb_dims,
                    bsz,
                    num_head,
                    seq_len,
                    dim_head);
        }
        phi::DenseTensor *tmp_padding_offset_tensor =
            encoder_remove_padding ? &padding_offset_tensor : nullptr;
        if(FLAGS_use_cutlass_fmha){
          fmha_compute.ComputeForwardWithCutlassFMHA(cache_kv,
                                                    src_mask,
                                                    tmp_padding_offset_tensor,
                                                    sequence_lengths,
                                                    &q_transpose_out,
                                                    &kv_transpose_out,
                                                    cache_kv_out,
                                                    &qk_out,
                                                    nullptr,
                                                    &softmax_out,
                                                    &attn_dropout_mask_out,
                                                    &attn_dropout_out,
                                                    &qktv_out,
                                                    &fmha_out,
                                                    token_num);
        } else {
          fmha_compute.ComputeForwardWithoutTranspose(cache_kv,
                                                    src_mask,
                                                    tmp_padding_offset_tensor,
                                                    &q_transpose_out,
                                                    &kv_transpose_out,
                                                    cache_kv_out,
                                                    &qk_out,
                                                    nullptr,
                                                    &softmax_out,
                                                    &attn_dropout_mask_out,
                                                    &attn_dropout_out,
                                                    &qktv_out,
                                                    &fmha_out,
                                                    token_num);
        }
        
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step3";
      VLOG(0) << "fmha_out:" << fmha_out;
#endif
      VLOG(5)<<"Doing out_linear gemm, mnk:"<<token_num<<", "<<dim_embed<<", "<<hidden_size;
      if (custom_comm) {
        custom_comm->SwapInput(buf1);
      }

      out_linear_compute.ComputeForward(
          out_linear_weights[i], &fmha_out, nullptr, buf1, nullptr, true);

      if (custom_comm) {
        *buf1 = custom_comm->AllReduce();
      } else {
        AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
      }
      
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step4";
#endif

      // step5. ln(residual + dropout(input + bias))
      auto *ln_scale_data = ffn_ln_scales[i]->data<T>();
      phi::ResidualAddRmsNormWrapper<T, phi::GPUContext>(
        dev_ctx, 
        buf1->data<T>(),
        x_data,
        ln_scale_data,
        epsilon, 
        token_num, 
        dim_embed,  
        residual_out_data,
        buf1->data<T>());

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step5";
      VLOG(0) << "ffn1_input:" << *buf1;
#endif

      // step6. ffn matmul1
      VLOG(5)<<"Doing ffn1 gemm, mnk:"<<token_num<<", "<<dim_ffn<<", "<<dim_embed;
      ffn1_glu_helper.Compute(buf1,
                              ffn1_weights[i],
                              nullptr,
                              &ffn1_out,
                              &ffn1_dropout_out);
      
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step6";
      VLOG(0) << "FFN1 out:" << ffn1_dropout_out;
#endif

      // step8. ffn2 matmul
      VLOG(5)<<"Doing ffn2 gemm, mnk:"<<token_num<<", "<<dim_embed<<", "<<dim_ffn;
      if (custom_comm) {
        custom_comm->SwapInput(buf1);
      }
      ffn2_linear_compute.ComputeForward(
          ffn2_weights[i], &ffn1_dropout_out, nullptr, buf1, nullptr, true);
    
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step7";
      VLOG(0) << "ffn2_out:" << *buf1;
#endif

      VLOG(4) << "MPAllReduce 4: " << buf1->numel();
      if (custom_comm) {
        *buf1 = custom_comm->AllReduce();
      } else {
        AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
      }
      
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step8.1";
      VLOG(0) << "ffn2_out_rd:" << *buf1;
#endif

      // step9. residual bias
      if (i < layers - 1) {
        auto *ln_scale_data = ln_scales[i + 1]->data<T>();
        phi::ResidualAddRmsNormWrapper<T, phi::GPUContext>(
          dev_ctx, 
          buf1->data<T>(),
          residual_out_data,
          ln_scale_data, 
          epsilon, 
          token_num, 
          dim_embed, 
          buf1->data<T>(),
          buf0->data<T>());
      } else {
        ffn2_fused_dropout_helper.ResidualDropoutBias(
            dev_ctx,
            buf1->data<T>(),
            residual_out_data,
            nullptr, 
            buf1->data<T>(),
            dropout_mask_out_data);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step9";
      VLOG(0) << "residual_out:" << *buf1;
#endif
      x_data = buf1->data<T>();
      std::swap(buf0, buf1);
    }
    if (encoder_remove_padding) {
      InvokeRebuildPadding(dev_ctx,
                            from_data,
                            buf0->data<T>(),
                            padding_offset_data,
                            token_num,
                            dim_embed);
    }
  }
};



}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_llama,
                        ops::FusedLLAMAOpKernel<plat::bfloat16>,
                        ops::FusedLLAMAOpKernel<plat::float16>,
                        ops::FusedLLAMAOpKernel<float>
                        );
