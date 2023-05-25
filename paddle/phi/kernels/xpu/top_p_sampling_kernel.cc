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

#include "paddle/phi/kernels/top_p_sampling_kernel.h"
#include "xpu/refactor/nn_customization.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
namespace phi {

static inline void generate_rand(float* res, const uint64_t seed) {
  // std::random_device seed;
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<> distrib(0.0, 1.0);
  *res = distrib(engine);
}

template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& ps,
                        int random_seed,
                        DenseTensor* out,
                        DenseTensor* ids) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* ps_ptr = reinterpret_cast<const XPUType*>(ps.data<T>());
  XPUType* out_ptr = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));
  int64_t* ids_ptr = dev_ctx.template Alloc<int64_t>(ids);
  auto x_dims = x.dims();
  int bs = x_dims[0];
  int vocab_size = x_dims[1];
  int p_num = ps.numel();

  PADDLE_ENFORCE_EQ(
      p_num,
      bs,
      phi::errors::PreconditionNotMet(
          "Expected bs == p_num, but got bs=%d, p_num=%d.", bs, p_num));

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  std::vector<float> seed_vec(bs, 1);
  srand((unsigned int)(time(NULL)));
  for (int i = 0; i < bs; i++) {
    generate_rand(seed_vec.data() + i, rand());
  }
  float* rand_coeff_xpu = RAII_GUARD.alloc<float>(seed_vec.size());
  memory_utils::Copy(x.place(),
                     rand_coeff_xpu,
                     phi::CPUPlace(),
                     seed_vec.data(),
                     seed_vec.size());

  int r = xpu::top_p_sampling<XPUType, int64_t>(dev_ctx.x_context(),
                                                x_ptr,
                                                ps_ptr,
                                                rand_coeff_xpu,
                                                ids_ptr,
                                                bs,
                                                vocab_size,
                                                out_ptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "top_p_sampling");

  /*
  int* ids_int32_ptr = RAII_GUARD.alloc_l3_or_gm<int>(ids->numel());
  int r = xpu::sorted_topk<XPUType, int>(dev_ctx.x_context(),
      x_ptr, out_ptr, ids_int32_ptr, x.dims()[0], x.dims()[1], 1);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sorted_topk");
  r = xpu::cast<int, int64_t>(dev_ctx.x_context(),
      ids_int32_ptr, ids_ptr, ids->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  */
}

}  // namespace phi

PD_REGISTER_KERNEL(top_p_sampling,
                   XPU,
                   ALL_LAYOUT,
                   phi::TopPSamplingKernel,
                   float,
                   phi::dtype::float16) {}
