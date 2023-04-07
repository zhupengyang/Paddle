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
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

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
namespace patterns {

struct Conv2dXPUPattern : public PatternBase {
  Conv2dXPUPattern(PDPattern* pattern,
                   const std::string& name_scope,
                   const std::string& conv_type,
                   const std::string& act_type,
                   bool with_bias,
                   bool with_bn,
                   bool with_branch);
  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(ew_bias_add);
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(ew_branch_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(ew_bias_add_y);
  PATTERN_DECL_NODE(ew_bias_add_out);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_var);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(bn_var_out);
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_saved_var);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(ew_branch_add_in);
  PATTERN_DECL_NODE(ew_branch_add_out);
  PATTERN_DECL_NODE(act_out);

 private:
  std::string conv_type_;
  std::string act_type_;
  bool with_bias_{false};
  bool with_bn_{false};
  bool with_branch_{false};
};

Conv2dXPUPattern::Conv2dXPUPattern(PDPattern* pattern,
                                   const std::string& name_scope,
                                   const std::string& conv_type,
                                   const std::string& act_type,
                                   bool with_bias,
                                   bool with_bn,
                                   bool with_branch)
    : PatternBase(pattern, name_scope, name_scope),
      conv_type_(conv_type),
      act_type_(act_type),
      with_bias_(with_bias),
      with_bn_(with_bn),
      with_branch_(with_branch) {
  auto conv = pattern->NewNode(conv_repr())->assert_is_op(conv_type_);
  auto input = pattern->NewNode(input_repr())
                   ->assert_is_op_input(conv_type_, "Input")
                   ->AsInput();
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input(conv_type_, "Filter")
                         ->AsInput();
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output(conv_type_, "Output")
                      ->assert_var_not_persistable();
  conv->LinksFrom({input, conv_filter}).LinksTo({conv_out});
  // ew_bias_add op
  PDNode* ew_bias_add = nullptr;
  PDNode* ew_bias_add_y = nullptr;
  PDNode* ew_bias_add_out = nullptr;
  if (with_bias_) {
    conv_out->assert_is_op_input("elementwise_add", "X");
    ew_bias_add_y = pattern->NewNode(ew_bias_add_y_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var()
                        ->assert_has_n_outputs(1);
    ew_bias_add =
        pattern->NewNode(ew_bias_add_repr())->assert_is_op("elementwise_add");
    ew_bias_add_out = pattern->NewNode(ew_bias_add_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out");
    ew_bias_add->LinksFrom({conv_out, ew_bias_add_y})
        .LinksTo({ew_bias_add_out});
  } else {
    ew_bias_add_out = conv_out;
  }
  // batch_norm op
  PDNode* bn = nullptr;
  PDNode* bn_bias = nullptr;
  PDNode* bn_mean = nullptr;
  PDNode* bn_scale = nullptr;
  PDNode* bn_var = nullptr;
  PDNode* bn_out = nullptr;
  PDNode* bn_mean_out = nullptr;
  PDNode* bn_saved_mean = nullptr;
  PDNode* bn_var_out = nullptr;
  PDNode* bn_saved_var = nullptr;
  if (with_bn_) {
    ew_bias_add_out->assert_is_op_input("batch_norm", "X");
    bn_bias = pattern->NewNode(bn_bias_repr())
                  ->assert_is_op_input("batch_norm", "Bias")
                  ->assert_has_n_outputs(1);
    bn_mean = pattern->NewNode(bn_mean_repr())
                  ->assert_is_op_input("batch_norm", "Mean")
                  ->assert_has_n_outputs(1);
    bn_scale = pattern->NewNode(bn_scale_repr())
                   ->assert_is_op_input("batch_norm", "Scale")
                   ->assert_has_n_outputs(1);
    bn_var = pattern->NewNode(bn_var_repr())
                 ->assert_is_op_input("batch_norm", "Variance")
                 ->assert_has_n_outputs(1);
    bn = pattern->NewNode(bn_repr())->assert_is_op("batch_norm");
    bn_out =
        pattern->NewNode(bn_out_repr())->assert_is_op_output("batch_norm", "Y");
    bn_mean_out = pattern->NewNode(bn_mean_out_repr())
                      ->assert_is_op_output("batch_norm", "MeanOut");
    bn_saved_mean = pattern->NewNode(bn_saved_mean_repr())
                        ->assert_is_op_output("batch_norm", "SavedMean");
    bn_var_out = pattern->NewNode(bn_var_out_repr())
                     ->assert_is_op_output("batch_norm", "VarianceOut");
    bn_saved_var = pattern->NewNode(bn_saved_var_repr())
                       ->assert_is_op_output("batch_norm", "SavedVariance");
    bn->LinksFrom({ew_bias_add_out, bn_bias, bn_mean, bn_scale, bn_var})
        .LinksTo(
            {bn_out, bn_mean_out, bn_var_out, bn_saved_mean, bn_saved_var});
  } else {
    bn_out = ew_bias_add_out;
  }
  // ew_branch_add op
  PDNode* ew_branch_add = nullptr;
  PDNode* ew_branch_add_in = nullptr;
  PDNode* ew_branch_add_out = nullptr;
  if (with_branch_) {
    ew_branch_add_in = pattern->NewNode(ew_branch_add_in_repr())->AsInput();
    ew_branch_add =
        pattern->NewNode(ew_branch_add_repr())->assert_is_op("elementwise_add");
    ew_branch_add_out = pattern->NewNode(ew_branch_add_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out");
    ew_branch_add->LinksFrom({bn_out, ew_branch_add_in})
        .LinksTo({ew_branch_add_out});
  } else {
    ew_branch_add_out = bn_out;
  }
  // act op
  PDNode* act = nullptr;
  PDNode* act_out = nullptr;
  if (!act_type_.empty()) {
    ew_branch_add_out->assert_is_op_input(act_type_, "X")->AsIntermediate();
    act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
    act_out = pattern->NewNode(act_out_repr())
                  ->assert_is_op_output(act_type_, "Out")
                  ->assert_var_not_persistable();
    act->LinksFrom({ew_branch_add_out}).LinksTo({act_out});
  }
}

}  // namespace patterns

/*
fuse conv2d block in resnet50-like model to xpu_conv2d op
For example:
graph[1]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
                 elementwise_add -----conv_Bias
                      |
                      |
                 batch_norm ------in_Bias
                      |
                      |
                     act
                      |
                      |
                    out_Out
------------------------------------------------------
graph[2]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
                 batch_norm ------in_Bias
                      |
                      |
                    out_Out
------------------------------------------------------
graph[3]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
       in_X       batch_norm ------in_Bias
            \         |
              \       |
               elementwise_add
                      |
                      |
                     act
                      |
                      |
                    out_Out
------------------------------------------------------
graph[4]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
               elementwise_add ------in_Bias
                      |
                      |
                     act
                      |
                      |
                    out_Out
------------------------------------------------------
After the pass is applied:
                    in_Input
       in_Filter      |     in_FilterMax
                 \    |    /
                  \   |   /
  in_Branch ------- __xpu__conv2d ------ in_Bias
                       |    \
                       |     \
                       |      out_OutputMax
                    out_Output
*/
class Conv2dXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& conv_type,
                const std::string& act_type,
                bool with_bias,
                bool with_bn,
                bool with_branch) const;

  const std::string name_scope_{"conv2d_xpu_fuse_pass"};
};

void Conv2dXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto conv_type : {"conv2d", "depthwise_conv2d"}) {
    for (auto with_bias : {true, false}) {
      for (auto with_bn : {true, false}) {
        for (auto with_branch : {true, false}) {
          for (auto act_type : {
                   "relu",
                   "sigmoid",
                   "tanh",
                   "gelu",
                   "leaky_relu",
                   "hard_swish",
                   "hard_sigmoid",
                   "relu6",
                   "swish",
                   "",
               }) {
            found_subgraph_count += ApplyImpl(
                graph, conv_type, act_type, with_bias, with_bn, with_branch);
          }
        }
      }
    }
  }
  AddStatis(found_subgraph_count);
}

static void PrepareConv2dXPUBias(Graph* graph,
                                 Scope* scope,
                                 const std::string& bias_name,
                                 const phi::DenseTensor& bias_new_t,
                                 Node** bias_new) {
  size_t bias_new_hash = HashTensor<float>(bias_new_t);
  std::string pre_name = GetPrefixWithoutHash(bias_name);
  std::string bias_new_name = pre_name + "_#" + std::to_string(bias_new_hash);
  *bias_new = FindNodeWithName(graph, bias_new_name);
  if (*bias_new == nullptr) {
    VarDesc bias_new_desc(bias_new_name);
    bias_new_desc.SetPersistable(true);
    bias_new_desc.SetShape(vectorize(bias_new_t.dims()));
    bias_new_desc.SetDataType(
        framework::TransToProtoVarType(bias_new_t.dtype()));
    *bias_new = graph->CreateVarNode(&bias_new_desc);
    Assign(bias_new_t,
           scope->Var(bias_new_name)->GetMutable<phi::DenseTensor>());
  }
}

static void PrepareConv2dXPUInt16Filter(Graph* graph,
                                        Scope* scope,
                                        const std::string& filter_name,
                                        const phi::DenseTensor& filter_t_fp32,
                                        Node** filter_new,
                                        Node** filter_max) {
  phi::DenseTensor filter_new_t;
  Assign(filter_t_fp32, &filter_new_t);
  phi::DenseTensor filter_max_t;
  PrepareWeight<int16_t>(&filter_new_t, &filter_max_t, false);

  size_t filter_new_t_hash = HashTensor<int16_t>(filter_new_t);
  size_t filter_max_t_hash = HashTensor<float>(filter_max_t);
  std::string pre_name = GetPrefixWithoutHash(filter_name);
  std::string filter_new_name =
      pre_name + "_#" + std::to_string(filter_new_t_hash);
  std::string filter_max_name =
      pre_name + "_max_#" + std::to_string(filter_max_t_hash);
  *filter_new = FindNodeWithName(graph, filter_new_name);
  if (*filter_new == nullptr) {
    VarDesc filter_new_desc(filter_new_name);
    filter_new_desc.SetPersistable(true);
    filter_new_desc.SetShape(vectorize(filter_new_t.dims()));
    filter_new_desc.SetDataType(
        framework::TransToProtoVarType(filter_new_t.dtype()));
    *filter_new = graph->CreateVarNode(&filter_new_desc);
    VarDesc filter_max_desc(filter_max_name);
    filter_max_desc.SetPersistable(true);
    filter_max_desc.SetShape(vectorize(filter_max_t.dims()));
    filter_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    *filter_max = graph->CreateVarNode(&filter_max_desc);

    auto* filter_new_var = scope->FindVar(filter_new_name);
    if (filter_new_var == nullptr) {
      Assign(filter_new_t,
             scope->Var(filter_new_name)->GetMutable<phi::DenseTensor>());
      Assign(filter_max_t,
             scope->Var(filter_max_name)->GetMutable<phi::DenseTensor>());
    } else {
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(filter_max_name),
          platform::errors::Fatal(
              "dst_max(%s) variable should not be nullptr if dst(%s) "
              "variable is exist. (src_name is %s)",
              filter_max_name,
              filter_new_name,
              filter_name));
    }
  } else {
    *filter_max = FindNodeWithName(graph, filter_max_name);
    PADDLE_ENFORCE_NOT_NULL(
        *filter_max,
        platform::errors::Fatal(
            "dst_max(%s) variable should not be nullptr if dst(%s) "
            "variable is exist. (src_name is %s)",
            filter_max_name,
            filter_new_name,
            filter_name));
  }
}

static void PrepareWeightOneValue(Graph* graph,
                                  Scope* scope,
                                  Node** w_one_value) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  const auto& dev_ctxs = pool.device_contexts();
  auto place = phi::XPUPlace();  // xpu:0
  for (auto it = dev_ctxs.begin(); it != dev_ctxs.end(); it++) {
    if (it->first.GetType() == phi::AllocationType::XPU) {  // maybe xpu:1
      place = it->first;
    }
  }
  phi::XPUContext* xpu_ctx = static_cast<phi::XPUContext*>(pool.Get(place));
  int max_ptr_size = xpu_ctx->x_context()->max_ptr_size();

  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  phi::DenseTensor w_one_value_t;
  w_one_value_t.Resize({static_cast<int64_t>(max_ptr_size)});
  float* one_value = cpu_ctx->Alloc<float>(&w_one_value_t);
  for (int i = 0; i < max_ptr_size; i++) {
    one_value[i] = 1.;
  }

  size_t w_one_value_t_hash = HashTensor<float>(w_one_value_t);
  std::string pre_name = "_conv2d_xpu_w_one_value";
  std::string w_one_value_name =
      pre_name + "_#" + std::to_string(w_one_value_t_hash);
  *w_one_value = FindNodeWithName(graph, w_one_value_name);
  if (*w_one_value == nullptr) {
    VarDesc w_one_value_desc(w_one_value_name);
    w_one_value_desc.SetPersistable(true);
    w_one_value_desc.SetShape(vectorize(w_one_value_t.dims()));
    w_one_value_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    *w_one_value = graph->CreateVarNode(&w_one_value_desc);
    Assign(w_one_value_t,
           scope->Var(w_one_value_name)->GetMutable<phi::DenseTensor>());
  }
}

int Conv2dXPUFusePass::ApplyImpl(ir::Graph* graph,
                                 const std::string& conv_type,
                                 const std::string& act_type,
                                 bool with_bias,
                                 bool with_bn,
                                 bool with_branch) const {
  GraphPatternDetector gpd;
  patterns::Conv2dXPUPattern pattern(gpd.mutable_pattern(),
                                     name_scope_,
                                     conv_type,
                                     act_type,
                                     with_bias,
                                     with_bn,
                                     with_branch);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle Conv2dXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(conv);
    GET_IR_NODE(ew_bias_add);
    GET_IR_NODE(bn);
    GET_IR_NODE(ew_branch_add);
    GET_IR_NODE(act);
    // declare variable node's name
    GET_IR_NODE(input);
    GET_IR_NODE(conv_filter);
    GET_IR_NODE(conv_out);
    GET_IR_NODE(ew_bias_add_y);
    GET_IR_NODE(ew_bias_add_out);
    GET_IR_NODE(bn_bias);
    GET_IR_NODE(bn_mean);
    GET_IR_NODE(bn_scale);
    GET_IR_NODE(bn_var);
    GET_IR_NODE(bn_out);
    GET_IR_NODE(bn_var_out);
    GET_IR_NODE(bn_mean_out);
    GET_IR_NODE(bn_saved_var);
    GET_IR_NODE(bn_saved_mean);
    GET_IR_NODE(ew_branch_add_in);
    GET_IR_NODE(ew_branch_add_out);
    GET_IR_NODE(act_out);
    // "ew_branch_add" may be include in two subgraphs. If we find
    // "ew_branch_add" again, we drop the fuse.
    if (with_branch && graph->Nodes().count(ew_branch_add) == 0) {
      return;
    }

    auto* block = conv->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    bool enable_int8 =
        graph->Has("enable_int8") && graph->Get<bool>("enable_int8");
    bool has_bias = with_bias || with_bn;
    auto* filter_t =
        scope->FindVar(conv_filter->Name())->GetMutable<phi::DenseTensor>();
    auto filter_dims = filter_t->dims();
    int kernel_dtype = proto::VarType::Type::VarType_Type_FP32;
    if (filter_t->dtype() == phi::DataType::FLOAT16) {
      kernel_dtype = proto::VarType::Type::VarType_Type_FP16;
    } else if (filter_t->dtype() == phi::DataType::INT8) {
      kernel_dtype = proto::VarType::Type::VarType_Type_INT8;
    }

    phi::DenseTensor bias_t;
    bias_t.Resize({filter_dims[0]});
    memset(cpu_ctx->Alloc<float>(&bias_t), 0, filter_dims[0] * sizeof(float));
    if (with_bias) {
      auto* ew_bias_add_y_t =
          scope->FindVar(ew_bias_add_y->Name())->GetMutable<phi::DenseTensor>();
      auto ew_bias_add_y_dims = ew_bias_add_y_t->dims();
      PADDLE_ENFORCE_EQ(filter_dims[0],
                        ew_bias_add_y_dims[0],
                        platform::errors::InvalidArgument(
                            "the shape[%d] of elewise bias tensor "
                            "must equal out_channel[%d] of conv",
                            ew_bias_add_y_dims[0],
                            filter_dims[0]));
      Assign(*ew_bias_add_y_t, &bias_t);
    }
    CastToFp32(&bias_t);

    // recompute bias and weight for conv2d_xpu op
    Node* filter_new = nullptr;
    Node* filter_max = nullptr;
    if (with_bn) {
      // fuse bn
      float* bn_scale_data = scope->Var(bn_scale->Name())
                                 ->GetMutable<phi::DenseTensor>()
                                 ->data<float>();
      float* bn_bias_data = scope->Var(bn_bias->Name())
                                ->GetMutable<phi::DenseTensor>()
                                ->data<float>();
      float* bn_mean_data = scope->Var(bn_mean->Name())
                                ->GetMutable<phi::DenseTensor>()
                                ->data<float>();
      float* bn_var_data = scope->Var(bn_var->Name())
                               ->GetMutable<phi::DenseTensor>()
                               ->data<float>();
      float epsilon = bn->Op()->GetAttrIfExists<float>("epsilon");

      // recompute bias
      auto* bias_data = bias_t.data<float>();
      for (int64_t i = 0; i < filter_dims[0]; i++) {
        float trans_scale = bn_scale_data[i] / sqrtf(bn_var_data[i] + epsilon);
        bias_data[i] =
            (bias_data[i] - bn_mean_data[i]) * trans_scale + bn_bias_data[i];
      }

      // recompute filter
      int64_t filter_step = filter_t->numel() / filter_dims[0];
      if (!enable_int8) {
        // float32/float16 filter
        phi::DenseTensor filter_t_new;
        Assign(*filter_t, &filter_t_new);
        CastToFp32(&filter_t_new);
        float* filter_data = filter_t_new.data<float>();
        for (int64_t i = 0; i < filter_dims[0]; i++) {
          float trans_scale =
              bn_scale_data[i] / sqrtf(bn_var_data[i] + epsilon);
          for (int64_t j = 0; j < filter_step; j++) {
            filter_data[i * filter_step + j] *= trans_scale;
          }
        }
        PrepareConv2dXPUInt16Filter(graph,
                                    scope,
                                    conv_filter->Name(),
                                    filter_t_new,
                                    &filter_new,
                                    &filter_max);

      } else {
        // int8 filter
        std::vector<float> max_value = PADDLE_GET_CONST(
            std::vector<float>, conv->Op()->GetAttr("Filter_max"));
        if (max_value.size() == 1) {
          max_value = std::vector<float>(filter_dims[0], max_value[0]);
        }
        for (int64_t i = 0; i < filter_dims[0]; i++) {
          float trans_scale =
              bn_scale_data[i] / sqrtf(bn_var_data[i] + epsilon);
          max_value[i] *= trans_scale;
        }
        PrepareMax(graph, scope, conv_filter->Name(), max_value, &filter_max);
        filter_new = conv_filter;
      }
    } else {
      // not fuse bn
      if (!enable_int8) {
        // float32/float16 filter
        PrepareWeight<int16_t>(
            graph, scope, block, conv_filter, &filter_new, &filter_max, false);
      } else {
        // int8 filter
        std::vector<float> max_value = PADDLE_GET_CONST(
            std::vector<float>, conv->Op()->GetAttr("Filter_max"));
        if (max_value.size() == 1) {
          max_value = std::vector<float>(filter_dims[0], max_value[0]);
        }
        PrepareMax(graph, scope, conv_filter->Name(), max_value, &filter_max);
        filter_new = conv_filter;
      }
    }

    Node* w_one_value = nullptr;
    if (enable_int8) {
      PrepareWeightOneValue(graph, scope, &w_one_value);
    }

    Node* bias_new = nullptr;
    PrepareConv2dXPUBias(graph,
                         scope,
                         with_bias ? ew_bias_add_y->Name() : "",
                         bias_t,
                         &bias_new);

    Node* branch_max = nullptr;
    if (with_branch && enable_int8) {
      auto branch_name = ew_branch_add_in->Name();
      auto branch_max_value = GetMaxAttr(ew_branch_add->Op(), branch_name);
      PrepareMax(graph, scope, branch_name, branch_max_value, &branch_max);
    }

    Node* x_max = nullptr;
    if (enable_int8) {
      auto x_name = input->Name();
      auto x_max_value = PADDLE_GET_CONST(std::vector<float>,
                                          conv->Op()->GetAttr("Input_max"));
      PrepareMax(graph, scope, x_name, x_max_value, &x_max);
    }

    // output && output max
    Node* out_max = nullptr;
    std::string out_name;
    std::vector<float> out_max_value;
    if (!act_type.empty()) {
      out_name = act_out->Name();
      out_max_value = act->Op()->GetAttrIfExists<std::vector<float>>("Out_max");
    } else if (ew_branch_add) {
      out_name = ew_branch_add_out->Name();
      out_max_value =
          ew_branch_add->Op()->GetAttrIfExists<std::vector<float>>("Out_max");
    } else if (bn) {
      out_name = bn_out->Name();
      out_max_value = bn->Op()->GetAttrIfExists<std::vector<float>>("Y_max");
    } else if (ew_bias_add) {
      out_name = ew_bias_add_out->Name();
      out_max_value =
          ew_bias_add->Op()->GetAttrIfExists<std::vector<float>>("Out_max");
    } else {
      out_name = conv_out->Name();
      out_max_value =
          conv->Op()->GetAttrIfExists<std::vector<float>>("Output_max");
    }
    if (enable_int8) {
      PrepareMax(graph, scope, out_name, out_max_value, &out_max);
    } else {
      std::string out_max_name = out_name + "_max";
      VarDesc out_max_desc(out_max_name);
      out_max = graph->CreateVarNode(&out_max_desc);
    }

    // Generate conv2d_xpu op
    framework::OpDesc conv2d_xpu_op_desc(block);
    conv2d_xpu_op_desc.SetType("conv2d_xpu");
    conv2d_xpu_op_desc.SetInput("x", {input->Name()});
    conv2d_xpu_op_desc.SetInput("w", {filter_new->Name()});
    conv2d_xpu_op_desc.SetInput("w_max", {filter_max->Name()});
    conv2d_xpu_op_desc.SetOutput("out", {out_name});
    conv2d_xpu_op_desc.SetOutput("out_max", {out_max->Name()});
    if (enable_int8) {
      conv2d_xpu_op_desc.SetInput("x_max", {x_max->Name()});
      conv2d_xpu_op_desc.SetInput("w_one_value", {w_one_value->Name()});
    }
    if (has_bias) {
      conv2d_xpu_op_desc.SetInput("bias", {bias_new->Name()});
    }
    if (with_branch) {
      conv2d_xpu_op_desc.SetInput("branch", {ew_branch_add_in->Name()});
      if (enable_int8) {
        conv2d_xpu_op_desc.SetInput("branch_max", {branch_max->Name()});
      }
    }
    float act_param = 0.0f;
    if (act_type == "leaky_relu") {
      act_param = PADDLE_GET_CONST(float, act->Op()->GetAttr("alpha"));
    } else if (act_type == "hard_sigmoid") {
      act_param = PADDLE_GET_CONST(float, act->Op()->GetAttr("slope"));
    }
    conv2d_xpu_op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    conv2d_xpu_op_desc.SetAttr("act_param", act_param);
    conv2d_xpu_op_desc.SetAttr("kernel_dtype", kernel_dtype);
    // "out_dtype" shoule be the same as input datatype.
    // If model is int8, "out_dtype" will be reset in "reset_out_dtype_pass".
    conv2d_xpu_op_desc.SetAttr("out_dtype",
                               static_cast<int>(input->Var()->GetDataType()));
    if (conv->Op()->HasAttr("padding_algorithm")) {
      conv2d_xpu_op_desc.SetAttr(
          "padding_algorithm",
          PADDLE_GET_CONST(std::string,
                           conv->Op()->GetAttr("padding_algorithm")));
    }
    auto conv_paddings =
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("paddings"));
    if (conv_paddings.size() == 2) {
      for (int i = 0; i < 2; i++) {
        int copy_pad = *(conv_paddings.begin() + 2 * i);
        conv_paddings.insert(conv_paddings.begin() + 2 * i + 1, copy_pad);
      }
    }
    PADDLE_ENFORCE_EQ(conv_paddings.size(),
                      4UL,
                      platform::errors::InvalidArgument(
                          "padding length should be 4, but received %d, ",
                          conv_paddings.size()));
    conv2d_xpu_op_desc.SetAttr("paddings", conv_paddings);
    conv2d_xpu_op_desc.SetAttr(
        "dilations",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("dilations")));
    conv2d_xpu_op_desc.SetAttr(
        "groups", PADDLE_GET_CONST(int, conv->Op()->GetAttr("groups")));
    conv2d_xpu_op_desc.SetAttr(
        "strides",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("strides")));
    auto* conv2d_xpu = graph->CreateOpNode(&conv2d_xpu_op_desc);
    IR_NODE_LINK_TO(input, conv2d_xpu);
    if (x_max) {
      IR_NODE_LINK_TO(x_max, conv2d_xpu);
    }
    IR_NODE_LINK_TO(filter_new, conv2d_xpu);
    IR_NODE_LINK_TO(filter_max, conv2d_xpu);
    if (w_one_value) {
      IR_NODE_LINK_TO(w_one_value, conv2d_xpu);
    }
    if (has_bias) {
      IR_NODE_LINK_TO(bias_new, conv2d_xpu);
    }
    if (ew_branch_add_in) {
      IR_NODE_LINK_TO(ew_branch_add_in, conv2d_xpu);
    }
    if (branch_max) {
      IR_NODE_LINK_TO(branch_max, conv2d_xpu);
    }
    if (act_out) {
      IR_NODE_LINK_TO(conv2d_xpu, act_out);
    } else if (ew_branch_add_out) {
      IR_NODE_LINK_TO(conv2d_xpu, ew_branch_add_out);
    } else if (bn_out) {
      IR_NODE_LINK_TO(conv2d_xpu, bn_out);
    } else if (ew_bias_add_out) {
      IR_NODE_LINK_TO(conv2d_xpu, ew_bias_add_out);
    } else {
      IR_NODE_LINK_TO(conv2d_xpu, conv_out);
    }
    IR_NODE_LINK_TO(conv2d_xpu, out_max);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {conv};
    if (act != nullptr) {
      delete_nodes.insert(act);
    }
    if (ew_branch_add != nullptr) {
      delete_nodes.insert(ew_branch_add);
    }
    if (bn != nullptr) {
      delete_nodes.insert(bn);
      delete_nodes.insert(bn_bias);
      delete_nodes.insert(bn_var);
      delete_nodes.insert(bn_mean);
      delete_nodes.insert(bn_scale);
      delete_nodes.insert(bn_var_out);
      delete_nodes.insert(bn_mean_out);
      delete_nodes.insert(bn_saved_var);
      delete_nodes.insert(bn_saved_mean);
    }
    if (ew_bias_add) {
      delete_nodes.insert(ew_bias_add);
      delete_nodes.insert(ew_bias_add_y);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv2d_xpu_fuse_pass, paddle::framework::ir::Conv2dXPUFusePass);

REGISTER_PASS_CAPABILITY(conv2d_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv2d_xpu", 0));
