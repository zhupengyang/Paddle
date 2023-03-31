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

#include "paddle/fluid/framework/ir/xpu/quant_xpu_pass.h"
#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct QuantDequantLinearPattern : public PatternBase {
  QuantDequantLinearPattern(PDPattern* pattern, const std::string& name_scope);

  PATTERN_DECL_NODE(quantize_linear_x);
  PATTERN_DECL_NODE(quantize_linear_scale);
  PATTERN_DECL_NODE(quantize_linear);
  PATTERN_DECL_NODE(quantize_linear_out);
  PATTERN_DECL_NODE(dequantize_linear);
  PATTERN_DECL_NODE(dequantize_linear_out);
};

QuantDequantLinearPattern::QuantDequantLinearPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto quantize_linear_x = pattern->NewNode(quantize_linear_x_repr())
                               ->assert_is_op_input("quantize_linear", "X");
  auto quantize_linear_scale =
      pattern->NewNode(quantize_linear_scale_repr())
          ->assert_is_op_input("quantize_linear", "Scale")
          ->assert_is_op_input("dequantize_linear", "Scale")
          ->assert_is_persistable_var();
  auto quantize_linear =
      pattern->NewNode(quantize_linear_repr())->assert_is_op("quantize_linear");
  auto quantize_linear_out = pattern->NewNode(quantize_linear_out_repr())
                                 ->assert_is_op_output("quantize_linear", "Y")
                                 ->assert_is_op_input("dequantize_linear", "X");
  auto dequantize_linear = pattern->NewNode(dequantize_linear_repr())
                               ->assert_is_op("dequantize_linear");
  auto dequantize_linear_out =
      pattern->NewNode(dequantize_linear_out_repr())
          ->assert_is_op_output("dequantize_linear", "Y");

  quantize_linear->LinksFrom({quantize_linear_x, quantize_linear_scale})
      .LinksTo({quantize_linear_out});
  dequantize_linear->LinksFrom({quantize_linear_out, quantize_linear_scale})
      .LinksTo({dequantize_linear_out});
}

}  // namespace patterns

void SetMaxAttr(OpDesc* op_desc,
                const std::vector<float>& max,
                const std::string& name) {
  auto inputs = op_desc->Inputs();
  for (auto input : inputs) {
    auto arg_name = input.first;
    auto var_names = input.second;
    if (std::count(var_names.begin(), var_names.end(), name) > 0) {
      op_desc->SetAttr(arg_name + "_max", max);
      break;
    }
  }
  auto outputs = op_desc->Outputs();
  for (auto output : outputs) {
    auto arg_name = output.first;
    auto var_names = output.second;
    if (std::count(var_names.begin(), var_names.end(), name) > 0) {
      op_desc->SetAttr(arg_name + "_max", max);
      break;
    }
  }
}

int QuantXPUPass::DeleteQuantDequantLinearPass(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::QuantDequantLinearPattern pattern(gpd.mutable_pattern(),
                                              name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DeleteQuantDequantLinearPass fuse";
    GET_IR_NODE(quantize_linear_x);
    GET_IR_NODE(quantize_linear_scale);
    GET_IR_NODE(quantize_linear);
    GET_IR_NODE(quantize_linear_out);
    GET_IR_NODE(dequantize_linear);
    GET_IR_NODE(dequantize_linear_out);
    auto* scope = param_scope();

    auto* max_tensor = scope->Var(quantize_linear_scale->Name())
                           ->GetMutable<phi::DenseTensor>();
    std::vector<float> max_value(max_tensor->numel());
    memcpy(max_value.data(),
           max_tensor->data<float>(),
           max_value.size() * sizeof(float));

    // Update next op
    auto dequant_out_name = dequantize_linear_out->Name();
    auto quant_x_name = quantize_linear_x->Name();
    for (auto* next_op : dequantize_linear_out->outputs) {
      auto* op_desc = next_op->Op();
      op_desc->RenameInput(dequant_out_name, quant_x_name);
      SetMaxAttr(op_desc, max_value, quant_x_name);
      IR_NODE_LINK_TO(quantize_linear_x, next_op);
    }

    // Update previous op
    for (auto* pre_op : quantize_linear_x->inputs) {
      auto* op_desc = pre_op->Op();
      SetMaxAttr(op_desc, max_value, quant_x_name);
    }

    std::unordered_set<const Node*> delete_nodes{
        quantize_linear, dequantize_linear, dequantize_linear_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

namespace patterns {

struct DequantLinearPattern : public PatternBase {
  DequantLinearPattern(PDPattern* pattern, const std::string& name_scope);

  PATTERN_DECL_NODE(dequantize_linear_x);
  PATTERN_DECL_NODE(dequantize_linear_scale);
  PATTERN_DECL_NODE(dequantize_linear);
  PATTERN_DECL_NODE(dequantize_linear_out);
};

DequantLinearPattern::DequantLinearPattern(PDPattern* pattern,
                                           const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto dequantize_linear_x = pattern->NewNode(dequantize_linear_x_repr())
                                 ->assert_is_op_input("dequantize_linear", "X")
                                 ->assert_is_persistable_var();
  auto dequantize_linear_scale =
      pattern->NewNode(dequantize_linear_scale_repr())
          ->assert_is_op_input("dequantize_linear", "Scale");
  auto dequantize_linear = pattern->NewNode(dequantize_linear_repr())
                               ->assert_is_op("dequantize_linear");
  auto dequantize_linear_out =
      pattern->NewNode(dequantize_linear_out_repr())
          ->assert_is_op_output("dequantize_linear", "Y");

  dequantize_linear->LinksFrom({dequantize_linear_x, dequantize_linear_scale})
      .LinksTo({dequantize_linear_out});
}

}  // namespace patterns

int QuantXPUPass::DeleteDequantLinearPass(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::DequantLinearPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DeleteDequantLinearPass fuse";
    GET_IR_NODE(dequantize_linear_x);
    GET_IR_NODE(dequantize_linear_scale);
    GET_IR_NODE(dequantize_linear);
    GET_IR_NODE(dequantize_linear_out);
    auto* scope = param_scope();

    auto* max_tensor = scope->Var(dequantize_linear_scale->Name())
                           ->GetMutable<phi::DenseTensor>();
    std::vector<float> max_value(max_tensor->numel());
    memcpy(max_value.data(),
           max_tensor->data<float>(),
           max_value.size() * sizeof(float));

    // Update next op
    auto dequant_out_name = dequantize_linear_out->Name();
    auto dequant_x_name = dequantize_linear_x->Name();
    for (auto* next_op : dequantize_linear_out->outputs) {
      auto* op_desc = next_op->Op();
      op_desc->RenameInput(dequant_out_name, dequant_x_name);
      SetMaxAttr(op_desc, max_value, dequant_x_name);
      IR_NODE_LINK_TO(dequantize_linear_x, next_op);
    }

    std::unordered_set<const Node*> delete_nodes{dequantize_linear,
                                                 dequantize_linear_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void QuantXPUPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Name() == "dequantize_linear") {
      if (graph->Has("enable_int8")) {
        graph->Erase("enable_int8");
      }
      graph->Set("enable_int8", new bool{true});
      break;
    }
  }

  int found_subgraph_count = DeleteQuantDequantLinearPass(graph);
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " quant_dequant_linear subgraph";
  }

  found_subgraph_count = DeleteDequantLinearPass(graph);
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " dequant_linear subgraph";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_xpu_pass, paddle::framework::ir::QuantXPUPass);

REGISTER_PASS_CAPABILITY(quant_xpu_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "quant_linear", 0))
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "dequant_linear", 0));
