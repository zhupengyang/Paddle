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
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace phi {
namespace fusion {

template <typename TX, typename TW, typename TOUT, typename Context>
void Conv2dXPUKernelImpl(const Context& ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& x_max,
                         const DenseTensor& w,
                         const DenseTensor& w_max,
                         const paddle::optional<DenseTensor>& w_one_value,
                         const paddle::optional<DenseTensor>& bias,
                         const paddle::optional<DenseTensor>& branch,
                         const paddle::optional<DenseTensor>& branch_max,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations,
                         const std::vector<int>& strides,
                         const std::string& padding_algorithm,
                         int groups,
                         int act_type,
                         float act_param,
                         DataType kernel_dtype,
                         DataType out_dtype,
                         DenseTensor* out,
                         DenseTensor* out_max) {
  using XPUTypeX = typename XPUTypeTrait<TX>::Type;
  using XPUTypeOut = typename XPUTypeTrait<TOUT>::Type;

  auto x_dims = x.dims();
  auto w_dims = w.dims();
  // update paddings and dilations accoring to padding_algorithm
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  DDim x_data_dims = phi::slice_ddim(x_dims, 2, x_dims.size());
  DDim w_data_dims = phi::slice_ddim(w_dims, 2, w_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(w_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                x_data_dims,
                                strides,
                                ksize);

  int batch = static_cast<int>(x_dims[0]);
  int in_c = static_cast<int>(x_dims[1]);
  int in_h = static_cast<int>(x_dims[2]);
  int in_w = static_cast<int>(x_dims[3]);
  int out_c = static_cast<int>(w_dims[0]);

  auto* x_data = reinterpret_cast<const XPUTypeX*>(x.data<TX>());
  const float* x_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* w_data = w.data<TW>();
  auto* w_max_data = w_max.data<float>();
  auto* w_one_value_data = w_one_value.get_ptr() == nullptr
                               ? nullptr
                               : w_one_value.get_ptr()->data<float>();
  bool per_channel = w_one_value_data != nullptr;
  const float* weight_max_data = per_channel ? w_one_value_data : w_max_data;
  const float* scale_max_data = per_channel ? w_max_data : nullptr;
  auto* branch_data =
      branch.get_ptr() == nullptr
          ? nullptr
          : reinterpret_cast<const XPUTypeOut*>(branch.get_ptr()->data<TOUT>());
  auto* branch_max_data =
      (branch_max.get_ptr() == nullptr || branch->dtype() != DataType::INT8)
          ? nullptr
          : branch_max.get_ptr()->data<float>();
  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto* out_data = reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<TOUT>(out));
  out_max->Resize({static_cast<int64_t>(ctx.x_context()->max_ptr_size())});
  auto* out_max_data = ctx.template Alloc<float>(out_max);

  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == xpu::Activation_t::LEAKY_RELU) {
    act.leaky_alpha = act_param;
  } else if (act_type == xpu::Activation_t::HARD_SIGMOID) {
    act.hard_sigmoid_slope = act_param;
  }
  int r = xpu::conv2d_fusion<XPUTypeX, TW, XPUTypeOut, TW>(  // TX/TW/TY/TGEMM
      /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
      /* const TX* x */ x_data,
      /* const TW* weight */ w_data,
      /* TY* out */ out_data,
      /* int64_t n */ batch,
      /* int64_t ic */ in_c,
      /* int64_t h */ in_h,
      /* int64_t w */ in_w,
      /* int64_t oc */ out_c,
      /* const std::vector<int>& ksize */ ksize,
      /* const std::vector<int>& strides */ strides,
      /* const std::vector<int>& paddings */ paddings_vec,
      /* const std::vector<int>& dilations */ dilations_vec,
      /* int64_t groups */ groups,
      /* const float* in_maxptr */ x_max_data,
      /* const float* filter_maxptr */ weight_max_data,
      /* float* out_maxptr */ out_max_data,
      /* bool is_nchw */ true,
      /* const float* bias */ bias_data,
      /* const TY* branch */ branch_data,
      /* const baidu::xpu::api::Activation_t& act */ act,
      /* const float* branch_maxptr */ branch_max_data,
      /* const float* scale */ scale_max_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_xpu");
}

template <typename T, typename Context>
void Conv2dXPUKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& x_max,
                     const DenseTensor& w,
                     const DenseTensor& w_max,
                     const paddle::optional<DenseTensor>& w_one_value,
                     const paddle::optional<DenseTensor>& bias,
                     const paddle::optional<DenseTensor>& branch,
                     const paddle::optional<DenseTensor>& branch_max,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const std::string& padding_algorithm,
                     int groups,
                     int act_type,
                     float act_param,
                     DataType kernel_dtype,
                     DataType out_dtype,
                     DenseTensor* out,
                     DenseTensor* out_max) {
  Conv2dXPUKernelImpl<T, int16_t, T, Context>(ctx,
                                              x,
                                              x_max,
                                              w,
                                              w_max,
                                              w_one_value,
                                              bias,
                                              branch,
                                              branch_max,
                                              paddings,
                                              dilations,
                                              strides,
                                              padding_algorithm,
                                              groups,
                                              act_type,
                                              act_param,
                                              kernel_dtype,
                                              out_dtype,
                                              out,
                                              out_max);
}

#define CONV2D_XPU_INT8_KERNEL(x_dtype_, out_dtype_)             \
  Conv2dXPUKernelImpl<x_dtype_, int8_t, out_dtype_, XPUContext>( \
      ctx,                                                       \
      x,                                                         \
      x_max,                                                     \
      w,                                                         \
      w_max,                                                     \
      w_one_value,                                               \
      bias,                                                      \
      branch,                                                    \
      branch_max,                                                \
      paddings,                                                  \
      dilations,                                                 \
      strides,                                                   \
      padding_algorithm,                                         \
      groups,                                                    \
      act_type,                                                  \
      act_param,                                                 \
      kernel_dtype,                                              \
      out_dtype,                                                 \
      out,                                                       \
      out_max);

template <>
void Conv2dXPUKernel<int8_t, XPUContext>(
    const XPUContext& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& x_max,
    const DenseTensor& w,
    const DenseTensor& w_max,
    const paddle::optional<DenseTensor>& w_one_value,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& branch,
    const paddle::optional<DenseTensor>& branch_max,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::string& padding_algorithm,
    int groups,
    int act_type,
    float act_param,
    DataType kernel_dtype,
    DataType out_dtype,
    DenseTensor* out,
    DenseTensor* out_max) {
  LOG(INFO) << "Conv2dXPUKernel, in_dtype: " << phi::DataTypeToString(x.type());
  LOG(INFO) << "Conv2dXPUKernel, kernel_dtype: "
            << phi::DataTypeToString(kernel_dtype);
  LOG(INFO) << "Conv2dXPUKernel, out_dtype: "
            << phi::DataTypeToString(out_dtype);

  auto x_dtype = x.type();
  if (x_dtype == DataType::INT8 && out_dtype == DataType::INT8) {
    CONV2D_XPU_INT8_KERNEL(int8_t, int8_t);
  } else if (x_dtype == DataType::INT8 && out_dtype == DataType::FLOAT32) {
    CONV2D_XPU_INT8_KERNEL(int8_t, float);
  } else if (x_dtype == DataType::FLOAT32 && out_dtype == DataType::INT8) {
    CONV2D_XPU_INT8_KERNEL(float, int8_t);
  } else if (x_dtype == DataType::FLOAT32 && out_dtype == DataType::FLOAT32) {
    CONV2D_XPU_INT8_KERNEL(float, float);
  } else if (x_dtype == DataType::INT8 && out_dtype == DataType::FLOAT16) {
    CONV2D_XPU_INT8_KERNEL(int8_t, phi::dtype::float16);
  } else if (x_dtype == DataType::FLOAT16 && out_dtype == DataType::INT8) {
    CONV2D_XPU_INT8_KERNEL(phi::dtype::float16, int8_t);
  } else if (x_dtype == DataType::FLOAT16 && out_dtype == DataType::FLOAT16) {
    CONV2D_XPU_INT8_KERNEL(phi::dtype::float16, phi::dtype::float16);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Expected type of Input(x)/Output(out) should be int8/float16/float32, "
        "but received type of Input(x) is %s, type of Output(out) is %s.",
        DataTypeToString(x_dtype),
        DataTypeToString(out_dtype)));
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::Conv2dXPUKernel,
                   float,
                   phi::dtype::float16,
                   int8_t) {}
