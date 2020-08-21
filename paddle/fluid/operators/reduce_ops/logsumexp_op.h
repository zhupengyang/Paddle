// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <vector>
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

#define LOGSUMEXP_DIM(NDIM, RDIM)                                         \
  if (ndim == NDIM && rdim == RDIM) {                                     \
    LogsumexpFunctor<DeviceContext, OutT, NDIM, RDIM>(                    \
        context.template device_context<DeviceContext>(), *input, output, \
        dims, keep_dim);                                                  \
    return;                                                               \
  }

template <typename DeviceContext, typename T, size_t D, size_t R_D>
void LogsumexpFunctor(const DeviceContext& context,
                      const framework::Tensor& input, framework::Tensor* output,
                      const std::vector<int>& dims, bool keep_dim) {
  auto reduce_dim = Eigen::array<int, R_D>();
  for (size_t i = 0; i < dims.size(); i++) {
    reduce_dim[i] = dims[i];
  }

  auto reshape_dim = Eigen::array<int, D>();
  auto input_dim = input.dims();
  for (size_t i = 0; i < input_dim.size(); i++) {
    reshape_dim[i] = input_dim[i];
  }
  for (size_t i = 0; i < dims.size(); i++) {
    reshape_dim[dims[i]] = 1;
  }

  auto broadcast_dim = Eigen::array<int, D>();
  for (size_t i = 0; i < input_dim.size(); i++) {
    broadcast_dim[i] = 1;
  }
  for (size_t i = 0; i < dims.size(); i++) {
    broadcast_dim[dims[i]] = input_dim[dims[i]];
  }

  DDim out_dims = output->dims();
  if (keep_dim) {
    const int kDelFlag = -2;
    auto dims_vector = framework::vectorize(out_dims);
    for (size_t i = 0; i < dims_ref.size(); ++i) {
      dims_vector[dims_ref[i]] = kDelFlag;
    }
    dims_vector.erase(remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
                      dims_vector.end());
    out_dims = framework::make_ddim(dims_vector);
  }

  auto x = EigenTensor<T, D>::From(input);
  auto out = EigenTensor<T, (D - R_D)>::From(*output, out_dims);
  auto x_max = x.maximum(reshape_dim);
  out.device(place) = (x - x_max.reshape(reshape_dim).broadcast(broadcast_dim))
                          .exp()
                          .sum(reduce_dim)
                          .log() +
                      x_max;
}

template <typename DeviceContext, typename T>
class LogsumexpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    auto dims = context.Attr<std::vector<int>>("dim");
    bool keep_dim = context.Attr<bool>("keep_dim");

    const auto& input_dim_size = input->dims().size();
    if (dims.size() == input_dim_size) {
      reduce = True;
    }

    output->mutable_data<OutT>(context.GetPlace());
    if (reduce_all) {
      auto x = EigenVector<OutT>::Flatten(*input);
      auto out = EigenScalar<OutT>::Flatten(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();

      auto broadcast_dim = x.dimensions();
      auto reduce_dim = Eigen::array<int, 1>({{0}});
      auto x_dim = Eigen::array<int, 1>({{0}});
      auto x_max = x.maximum(reduce_dim);
      out.device(place) =
          (x - x_max.broadcast(broadcast_dim)).exp().sum(reduce_dim).log() +
          x_max;
    } else {
      int ndim = input_dim_size;
      int rdim = dims.size();
      LOGSUMEXP_DIM(6, 5)
      LOGSUMEXP_DIM(6, 4)
      LOGSUMEXP_DIM(6, 3)
      LOGSUMEXP_DIM(6, 2)
      LOGSUMEXP_DIM(6, 1)
      LOGSUMEXP_DIM(5, 4)
      LOGSUMEXP_DIM(5, 3)
      LOGSUMEXP_DIM(5, 2)
      LOGSUMEXP_DIM(5, 1)
      LOGSUMEXP_DIM(4, 3)
      LOGSUMEXP_DIM(4, 2)
      LOGSUMEXP_DIM(4, 1)
      LOGSUMEXP_DIM(3, 2)
      LOGSUMEXP_DIM(3, 1)
      LOGSUMEXP_DIM(2, 1)
    }
  }
}

struct LogsumexpGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = dy->broadcast(dim) * (*x - y->broadcast(dim)).exp();
  }
};

}  // namespace operators
}  // namespace paddle
