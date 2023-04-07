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

#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/backends/xpu/xpu_op_list.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class ResetOutDtypePass : public Pass {
 protected:
  void ApplyImpl(Graph* graph) const override;
};

static bool IsOpSupportInt8(const std::string& op_type) {
  return phi::backends::xpu::is_xpu_support_op(op_type, phi::DataType::INT8);
}

void ResetOutDtypePass::ApplyImpl(Graph* graph) const {
  if (!graph->Has("enable_int8") || !graph->Get<bool>("enable_int8")) return;

  int reset_count = 0;
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) continue;
    auto* op_desc = node->Op();
    if (!IsOpSupportInt8(op_desc->Type()) || !op_desc->HasAttr("out_dtype"))
      continue;

    auto out_node_names = op_desc->Output("out");
    std::vector<Node*> next_ops;
    for (auto* out_node : node->outputs) {
      if (std::count(out_node_names.begin(),
                     out_node_names.end(),
                     out_node->Name()) > 0) {
        auto out_next_nodes = out_node->outputs;
        next_ops.insert(
            next_ops.end(), out_next_nodes.begin(), out_next_nodes.end());
      }
    }

    bool all_next_ops_int8 = true;
    for (auto* next_op : next_ops) {
      if (!IsOpSupportInt8(next_op->Name())) {
        all_next_ops_int8 = false;
        break;
      }
    }
    if (all_next_ops_int8) {
      op_desc->SetAttr(
          "out_dtype",
          static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
      reset_count++;
    }
  }
  if (reset_count > 0) {
    LOG(INFO) << "--- reset " << reset_count << " ops' out_dtype attr.";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reset_out_dtype_pass, paddle::framework::ir::ResetOutDtypePass);
