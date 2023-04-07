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

// Convert ops(not support int8) to xpu_ops(support int8):
// For example: convert "pool2d" to "pool2d_xpu" (has x_max and out_max)
class ConvertQuantXPUOpsPass : public Pass {
 protected:
  void ApplyImpl(Graph* graph) const override;
};

static void PrepareInput(Graph* graph,
                         Scope* scope,
                         Node* op_node,
                         const std::string& old_arg_name,
                         const std::string& new_arg_name) {
  auto* op_desc = op_node->Op();
  auto in_name = op_desc->Input(old_arg_name)[0];
  op_desc->RemoveInput(old_arg_name);
  op_desc->SetInput(new_arg_name, {in_name});
  auto in_max_value =
      op_desc->GetAttrIfExists<std::vector<float>>(old_arg_name + "_max");
  Node* in_max = nullptr;
  PrepareMax(graph, scope, in_name, in_max_value, &in_max);
  op_desc->SetInput(new_arg_name + "_max", {in_max->Name()});
  IR_NODE_LINK_TO(in_max, op_node);
}

static void PrepareOutput(Graph* graph,
                          Scope* scope,
                          Node* op_node,
                          const std::string& old_arg_name,
                          const std::string& new_arg_name) {
  auto* op_desc = op_node->Op();
  auto out_name = op_desc->Output(old_arg_name)[0];
  op_desc->RemoveOutput(old_arg_name);
  op_desc->SetOutput(new_arg_name, {out_name});
  auto out_max_value =
      op_desc->GetAttrIfExists<std::vector<float>>(old_arg_name + "_max");
  Node* out_max = nullptr;
  PrepareMax(graph, scope, out_name, out_max_value, &out_max);
  op_desc->SetOutput(new_arg_name + "_max", {out_max->Name()});
  IR_NODE_LINK_TO(op_node, out_max);
}

static void ConvertPool2D(Graph* graph, Scope* scope, Node* op_node) {
  op_node->RenameOp("pool2d_xpu");
  PrepareInput(graph, scope, op_node, "X", "x");
  PrepareOutput(graph, scope, op_node, "Out", "out");
  auto op_desc = op_node->Op();
  op_desc->SetAttr("kernel_size",
                   op_desc->GetAttrIfExists<std::vector<int>>("ksize"));
}

void ConvertQuantXPUOpsPass::ApplyImpl(Graph* graph) const {
  if (!graph->Has("enable_int8") || !graph->Get<bool>("enable_int8")) return;

  int convert_count = 0;
  Scope& scope = graph->Get<framework::Scope>("__param_scope__");
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) continue;

    auto op_type = node->Name();
    if (op_type == "pool2d") {
      ConvertPool2D(graph, &scope, node);
      convert_count++;
    }
  }
  if (convert_count > 0) {
    LOG(INFO) << "--- convert " << convert_count << " ops to xpu_op.";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(convert_quant_xpu_ops_pass,
              paddle::framework::ir::ConvertQuantXPUOpsPass);
