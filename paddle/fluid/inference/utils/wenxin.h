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

#if defined(_WIN32)
#ifdef WENXIN_LIB
#define WENXIN_DECL __declspec(dllexport)
#else
#define WENXIN_DECL __declspec(dllimport)
#endif  // WENXIN_LIB
#else
#define WENXIN_DECL __attribute__((visibility("default")))
#endif  // _WIN32

namespace wenxin {

WENXIN_DECL std::string UnitStringDecode(const std::string& str, int thread_num, const std::string& parse = "");

} // namespace wenxin

