// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/memcpy_kernel.h"
#ifdef PADDLE_WITH_XPU_XFT
#include "models/fused_multi_transformer_gpt.h"
namespace xft = baidu::xpu::xft;
#endif

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedMultiTransformerDyquantXpuKernel(
    const Context& ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& ln_scale,
    const std::vector<const DenseTensor*>& ln_bias,
    const std::vector<const DenseTensor*>& qkvw,
    const std::vector<const DenseTensor*>& qkvw_scale,
    const std::vector<const DenseTensor*>& qkv_bias,
    const std::vector<const DenseTensor*>& out_linear_w,
    const std::vector<const DenseTensor*>& out_linear_wscale,
    const std::vector<const DenseTensor*>& out_linear_bias,
    const std::vector<const DenseTensor*>& ffn_ln_scale,
    const std::vector<const DenseTensor*>& ffn_ln_bias,
    const std::vector<const DenseTensor*>& ffn1_weight,
    const std::vector<const DenseTensor*>& ffn1_weight_scale,
    const std::vector<const DenseTensor*>& ffn1_bias,
    const std::vector<const DenseTensor*>& ffn2_weight,
    const std::vector<const DenseTensor*>& ffn2_weight_scale,
    const std::vector<const DenseTensor*>& ffn2_bias,
    const paddle::optional<std::vector<const DenseTensor*>>& cache_kv,
    const paddle::optional<std::vector<const DenseTensor*>>& pre_caches,
    const paddle::optional<DenseTensor>& rotary_pos_emb,
    const paddle::optional<DenseTensor>& time_step,
    const paddle::optional<DenseTensor>& seq_lengths,
    const paddle::optional<DenseTensor>& src_mask,
    const DenseTensor& max_buffer,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    bool interleaved_weight,
    const std::string& int8_gemm_method,
    int ring_id,
    DenseTensor* out,
    std::vector<DenseTensor*> cache_kv_out) {
#ifdef PADDLE_WITH_XPU_XFT
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(pre_layer_norm,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Only support pre_layer_norm = true at now."));
  PADDLE_ENFORCE_EQ(
      seq_lengths.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("seq_lengths not support at now."));
  PADDLE_ENFORCE_EQ(
      rotary_pos_emb.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("rotary_pos_emb not support at now."));
  // TODO(mayang02): remove this warning
  if (pre_caches.get_ptr() != nullptr) {
    LOG(WARNING) << "Only support pre_caches is zeros at now.";
  }
  PADDLE_ENFORCE_NE(
      src_mask.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("src_mask should not be nullptr."));
  PADDLE_ENFORCE_EQ(trans_qkvw,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Only support trans_qkvw == true at now."));
  PADDLE_ENFORCE_EQ("weight-only", int8_gemm_method,
      phi::errors::PreconditionNotMet("The int8_gemm_method attr must be weight-only"));

  int layers = qkvw.size();
  const auto x_dims = x.dims();
  int seq_len = x_dims[1];
  const auto qkv_w_dims = qkvw[0]->dims();
  int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];

  int time_step_value = -1;
  if (time_step) {
    PADDLE_ENFORCE_EQ(time_step.get_ptr()->place(),
                      phi::CPUPlace(),
                      phi::errors::PreconditionNotMet(
                          "The place of input(time_step) must be CPUPlace."));
    // cache_seq_len
    time_step_value = time_step.get_ptr()->data<int>()[0];
    PADDLE_ENFORCE_GT(
        time_step_value,
        0,
        phi::errors::PreconditionNotMet(
            "The value of time_step must > 0, but now is %d", time_step_value));
    PADDLE_ENFORCE_EQ(
        seq_len,
        1,
        phi::errors::PreconditionNotMet(
            "In decode stage, the seq_len of input must be 1, but now is %d",
            seq_len));
  }

  XPUTypeT* x_data = reinterpret_cast<XPUTypeT*>(const_cast<T*>(x.data<T>()));
  // The value of fake_max is one
  float* fake_wmax_data = const_cast<float*>(max_buffer.data<float>());
  int max_ptr_size = ctx.x_context()->max_ptr_size();
  float* cache_k_max_buf = const_cast<float*>(max_buffer.data<float>() + max_ptr_size);
  float* cache_v_max_buf = const_cast<float*>(max_buffer.data<float>() + max_ptr_size * (1 + layers));
  XPUTypeT* src_mask_data = reinterpret_cast<XPUTypeT*>(
      const_cast<T*>(src_mask.get_ptr()->data<T>()));
  auto* out_data = reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(out));
  auto src_mask_dims = src_mask.get_ptr()->dims();
  auto out_dims = out->dims();
  auto xft_x = xft::xftTensor<XPUTypeT, 3>(
      x_data, std::array<int64_t, 3>{x_dims[0], x_dims[1], x_dims[2]});
  auto xft_src_mask =
      xft::xftTensor<XPUTypeT, 4>(src_mask_data,
                               std::array<int64_t, 4>{src_mask_dims[0],
                                                      src_mask_dims[1],
                                                      src_mask_dims[2],
                                                      src_mask_dims[3]});
  auto xft_out = xft::xftTensor<XPUTypeT, 3>(
      out_data, std::array<int64_t, 3>{out_dims[0], out_dims[1], out_dims[2]});

  typedef int8_t TW;
  std::vector<xft::xftVec<float>> xft_ln_scale;
  std::vector<xft::xftVec<float>> xft_ln_bias;
  std::vector<xft::xftMat<TW>> xft_qkvw;
  std::vector<xft::xftVec<float>> xft_qkv_bias;
  std::vector<xft::xftMat<TW>> xft_out_linear_w;
  std::vector<xft::xftVec<float>> xft_out_linear_bias;
  std::vector<xft::xftVec<float>> xft_ffn_ln_scale;
  std::vector<xft::xftVec<float>> xft_ffn_ln_bias;
  std::vector<xft::xftMat<TW>> xft_ffn1_w;
  std::vector<xft::xftVec<float>> xft_ffn1_bias;
  std::vector<xft::xftMat<TW>> xft_ffn2_w;
  std::vector<xft::xftVec<float>> xft_ffn2_bias;
  std::vector<xft::xftTensor<XPUTypeT, 5>> xft_cache_kv;
  std::vector<xft::xftTensor<XPUTypeT, 5>> xft_cache_kv_out;
  std::vector<xft::xftTensor<XPUTypeT, 5>> xft_pre_cache;
  std::vector<xft::xftTensor<XPUTypeT, 4>> xft_cache_k;
  std::vector<xft::xftTensor<XPUTypeT, 4>> xft_cache_v;

  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    xft_ln_scale.emplace_back(const_cast<float*>(ln_scale[i]->data<float>()),
                              std::array<int64_t, 1>{ln_scale[i]->dims()[0]});
    xft_ln_bias.emplace_back(const_cast<float*>(ln_bias[i]->data<float>()),
                             std::array<int64_t, 1>{ln_bias[i]->dims()[0]});
    // step2. qkv
    auto qkvw_dims = qkvw[i]->dims();
    xft_qkvw.emplace_back(
        const_cast<TW*>(qkvw[i]->data<TW>()), fake_wmax_data,
        const_cast<float*>(qkvw_scale[i]->data<float>()),
        std::array<int64_t, 2>{qkvw_dims[0] * qkvw_dims[1] * qkvw_dims[2],
                               qkvw_dims[3]});
    auto qkvb_dims = qkv_bias[i]->dims();
    xft_qkv_bias.emplace_back(
        const_cast<float*>(qkv_bias[i]->data<float>()),
        std::array<int64_t, 1>{qkvb_dims[0] * qkvb_dims[1] * qkvb_dims[2]});
    // attn out
    auto outw_dims = out_linear_w[i]->dims();
    xft_out_linear_w.emplace_back(
        const_cast<TW*>(out_linear_w[i]->data<TW>()), fake_wmax_data,
        const_cast<float*>(out_linear_wscale[i]->data<float>()),
        std::array<int64_t, 2>{outw_dims[0], outw_dims[1]});
    xft_out_linear_bias.emplace_back(
        const_cast<float*>(out_linear_bias[i]->data<float>()),
        std::array<int64_t, 1>{out_linear_bias[i]->dims()[0]});
    // ffn ln
    xft_ffn_ln_scale.emplace_back(
        const_cast<float*>(ffn_ln_scale[i]->data<float>()),
        std::array<int64_t, 1>{ffn_ln_scale[i]->dims()[0]});
    xft_ffn_ln_bias.emplace_back(
        const_cast<float*>(ffn_ln_bias[i]->data<float>()),
        std::array<int64_t, 1>{ffn_ln_bias[i]->dims()[0]});
    // ffn1
    auto ffn1w_dims = ffn1_weight[i]->dims();
    xft_ffn1_w.emplace_back(
        const_cast<TW*>(ffn1_weight[i]->data<TW>()), fake_wmax_data,
        const_cast<float*>(ffn1_weight_scale[i]->data<float>()),
        std::array<int64_t, 2>{ffn1w_dims[0], ffn1w_dims[1]});
    xft_ffn1_bias.emplace_back(const_cast<float*>(ffn1_bias[i]->data<float>()),
                               std::array<int64_t, 1>{ffn1_bias[i]->dims()[0]});
    // ffn2
    auto ffn2w_dims = ffn2_weight[i]->dims();
    xft_ffn2_w.emplace_back(
        const_cast<TW*>(ffn2_weight[i]->data<TW>()), fake_wmax_data,
        const_cast<float*>(ffn2_weight_scale[i]->data<float>()),
        std::array<int64_t, 2>{ffn2w_dims[0], ffn2w_dims[1]});
    xft_ffn2_bias.emplace_back(const_cast<float*>(ffn2_bias[i]->data<float>()),
                               std::array<int64_t, 1>{ffn2_bias[i]->dims()[0]});
    // cache kv in
    if (time_step_value > 0) {
      auto cachekv_dims = cache_kv.get_ptr()->at(i)->dims();
      xft_cache_kv.emplace_back(reinterpret_cast<XPUTypeT*>(const_cast<T*>(
                                    cache_kv.get_ptr()->at(i)->data<T>())),
                                std::array<int64_t, 5>{cachekv_dims[0],
                                                       cachekv_dims[1],
                                                       cachekv_dims[2],
                                                       cachekv_dims[3],
                                                       cachekv_dims[4]});
    } else {
      auto precache_dims = pre_caches.get_ptr()->at(i)->dims();
      xft_pre_cache.emplace_back(reinterpret_cast<XPUTypeT*>(const_cast<T*>(
              pre_caches.get_ptr()->at(i)->data<T>())),
          std::array<int64_t, 5>{precache_dims[0], precache_dims[1],
            precache_dims[2], precache_dims[3], precache_dims[4]});
    }
    // cache kv out
    auto cachekv_out_dims = cache_kv_out[i]->dims();
    xft_cache_kv_out.emplace_back(
        reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(cache_kv_out[i])),
        std::array<int64_t, 5>{cachekv_out_dims[0],
                               cachekv_out_dims[1],
                               cachekv_out_dims[2],
                               cachekv_out_dims[3],
                               cachekv_out_dims[4]});
    if (time_step_value > 0) {
      PADDLE_ENFORCE_EQ(cache_kv.get_ptr()->at(i)->data<T>(),
          cache_kv_out[i]->data<T>(),
          phi::errors::PreconditionNotMet("The cache_kv and cache_kv_out should be in-place."));
    }
    XPUTypeT* curr_cache_kv_ptr = reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(cache_kv_out[i]));
    int64_t half_len = cache_kv_out[i]->numel() / 2;
    float* curr_cache_k_max = cache_k_max_buf + i * max_ptr_size;
    float* curr_cache_v_max = cache_v_max_buf + i * max_ptr_size;
    xft_cache_k.emplace_back(curr_cache_kv_ptr, curr_cache_k_max, std::array<int64_t, 4>{cachekv_out_dims[1],
        cachekv_out_dims[2], cachekv_out_dims[3], cachekv_out_dims[4]});
    xft_cache_v.emplace_back(curr_cache_kv_ptr + half_len, curr_cache_v_max, std::array<int64_t, 4>{
        cachekv_out_dims[1], cachekv_out_dims[2], cachekv_out_dims[3], cachekv_out_dims[4]});
  }

  xft::NlpParam param;
  param.num_layer = layers;
  param.n_head = num_head;
  param.size_per_head = dim_head;
  param.hidden_act = act_method;
  param.is_fuse_qkv = true;

  // TODO: need more elegant code
  const char* layout_env = std::getenv("ATTN_LAYOUT");
  std::string attn_layout;
  if (layout_env != nullptr) {
    attn_layout = std::string(layout_env);
  } else {
    attn_layout = "LBHD";
  }
  const auto rope_dims = rotary_pos_emb.get_ptr()->dims();
  float* rope_data = const_cast<float*>(rotary_pos_emb.get_ptr()->data<float>());
  auto xft_rotary_pos_emb = xft::xftTensor<float, 4>(rope_data, 
                                           std::array<int64_t, 4>{rope_dims[0],
                                           rope_dims[1],
                                           rope_dims[2],
                                           rope_dims[3]});

  int r = xft::fused_multi_transformer_gpt<XPUTypeT, TW, int8_t>(ctx.x_context(),
                                                          xft_x,
                                                          xft_pre_cache,
                                                          xft_src_mask,
                                                          xft_rotary_pos_emb,
                                                          xft_ln_scale,
                                                          xft_ln_bias,
                                                          xft_qkvw,
                                                          xft_qkv_bias,
                                                          xft_out_linear_w,
                                                          xft_out_linear_bias,
                                                          xft_ffn_ln_scale,
                                                          xft_ffn_ln_bias,
                                                          xft_ffn1_w,
                                                          xft_ffn1_bias,
                                                          xft_ffn2_w,
                                                          xft_ffn2_bias,
                                                          &xft_out,
                                                          xft_cache_k,
                                                          xft_cache_v,
                                                          param,
                                                          time_step_value,
                                                          nullptr,
                                                          attn_layout);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xft::fused_multi_transformer_gpt");
#else
  LOG(FATAL) << "fused_multi_transformer_dyquant_xpu is not supported since it's not "
                "compiled with XPU_XFT";
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multi_transformer_dyquant_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiTransformerDyquantXpuKernel,
                   phi::dtype::float16) {
  kernel->InputAt(20).SetBackend(phi::Backend::CPU);
}
