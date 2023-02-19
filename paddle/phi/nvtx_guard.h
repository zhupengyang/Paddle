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

#pragma once

#include <string>
#include "gflags/gflags.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/nvtx.h"
#endif

DECLARE_bool(enable_nvtx);

namespace phi {

class NVTXGuard {
 public:
  explicit NVTXGuard(const char *s) {
#ifdef PADDLE_WITH_CUDA
    need_pop_ = FLAGS_enable_nvtx;
    if (need_pop_) {
      phi::dynload::nvtxRangePushA(s);
    }
#endif
  }

  explicit NVTXGuard(const std::string &s) : NVTXGuard(s.c_str()) {}

  ~NVTXGuard() {
#ifdef PADDLE_WITH_CUDA
    if (need_pop_) {
      phi::dynload::nvtxRangePop();
    }
#endif
  }

 private:
  bool need_pop_;

 private:
  NVTXGuard(const NVTXGuard &) = delete;
  NVTXGuard(NVTXGuard &&) = delete;
  NVTXGuard &operator=(const NVTXGuard &) = delete;
  NVTXGuard &operator=(NVTXGuard &&) = delete;
};

}  // namespace phi
