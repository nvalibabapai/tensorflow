/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/optimizers/auto_mixed_precision.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {

NodeDef& AutoMixedPrecision::CreateNodeDefCastf2h(
      const string& name,
      const string& device,
      const string& input) {
    NodeDef& cast_def = *optimized_graph_->add_node();
    cast_def.set_op("Cast");
    cast_def.set_name(name);
    cast_def.set_device(device);
    cast_def.add_input(input);
    node_map_->AddNode(cast_def.name(), &cast_def);
    node_map_->AddOutput(NodeName(input), cast_def.name());
    cast_def.mutable_attr()->insert({"SrcT", attr_float_});
    cast_def.mutable_attr()->insert({"DstT", attr_half_});
    cast_def.mutable_attr()->insert({"Truncate", attr_false_});
    return cast_def;
}

NodeDef& AutoMixedPrecision::CreateNodeDefCasth2f(
      const string& name,
      const string& device,
      const string& input) {
    NodeDef& cast_def = *optimized_graph_->add_node();
    cast_def.set_op("Cast");
    cast_def.set_name(name);
    cast_def.set_device(device);
    cast_def.add_input(input);
    node_map_->AddNode(cast_def.name(), &cast_def);
    node_map_->AddOutput(NodeName(input), cast_def.name());
    cast_def.mutable_attr()->insert({"SrcT", attr_half_});
    cast_def.mutable_attr()->insert({"DstT", attr_float_});
    cast_def.mutable_attr()->insert({"Truncate", attr_false_});
    return cast_def;
}

Status AutoMixedPrecision::HandleUnaryOp(const string& op, const NodeDef& node) {
  if (node.op() == op) {
    std::vector<string> inputs;
    std::vector<string> control_inputs;
    for (auto& input : node.input()) {
      if (IsControlInput(input)) {
        control_inputs.push_back(input);
      } else {
        inputs.push_back(input);
      }
    }
    if (node.op() == "Reshape") {
      if (inputs.size() != 2) {
        return Status(error::INVALID_ARGUMENT,
                      "Expected 2 inputs for " + node.op());
      }
    } else if (node.op() == "FusedBatchNorm") {
      if (inputs.size() != 5) {
        return Status(error::INVALID_ARGUMENT,
                      "Expected 5 inputs for " + node.op());
      }
    } else {
      if (inputs.size() != 1) {
        return Status(error::INVALID_ARGUMENT,
                      "Expected 1 input for " + node.op());
      }
    }
    NodeDef* input_node = node_map_->GetNode(inputs[0]);
    if (input_node->op() != "Cast" ||
        input_node->attr().at("SrcT").type() != DT_HALF) {
      return Status::OK();
    }
    node_map_->RemoveOutput(NodeName(inputs[0]), node.name());
    NodeDef& cast_def = CreateNodeDefCastf2h(
        /*name=*/node.name() + "/Cast",
        /*device=*/node.device(),
        /*input=*/inputs[0]);

    NodeDef& half_def = *optimized_graph_->add_node();
    auto op = node.op();
    if (op == "FusedBatchNorm"){
      op = "FusedBatchNormV2";
    }
    half_def.set_op(op);
    half_def.set_name(node.name() + "_amp");
    half_def.set_device(node.device());
    half_def.add_input(cast_def.name());
    node_map_->AddNode(half_def.name(), &half_def);
    node_map_->AddOutput(cast_def.name(), half_def.name());
    if (node.op() == "Reshape" || node.op() == "FusedBatchNorm") {
      for (int i = 1; i < inputs.size(); ++i) {
        half_def.add_input(inputs[i]);
        node_map_->RemoveOutput(NodeName(inputs[i]), node.name());
        node_map_->AddOutput(NodeName(inputs[i]), half_def.name());
      }
    }
    *half_def.mutable_attr() = node.attr();
    (*half_def.mutable_attr())["T"] = attr_half_;
    if (node.op() == "FusedBatchNorm"){
      (*half_def.mutable_attr())["U"] = attr_float_;
    }
    for (auto& control_input : control_inputs) {
      half_def.add_input(control_input);
      node_map_->RemoveOutput(NodeName(control_input), node.name());
      node_map_->AddOutput(NodeName(control_input), half_def.name());
    }

    if (node.op() != "Shape") {
      NodeDef& cast2_def = CreateNodeDefCasth2f(
        /*name=*/node.name() + "/Cast_2",
        /*device=*/node.device(),
        /*input=*/half_def.name());

      auto consumers = node_map_->GetOutputs(node.name());
      for (auto consumer : consumers) {
        for (int i = 0; i < consumer->input_size(); ++i) {
          if (NodeName(consumer->input(i)) == node.name()) {
            if (NodePosition(consumer->input(i)) <= 0) {
              if (IsControlInput(consumer->input(i))) {
                *consumer->mutable_input(i) = AsControlDependency(cast2_def);
              } else {
                *consumer->mutable_input(i) = cast2_def.name();
              }
              node_map_->AddOutput(cast2_def.name(), consumer->name());
              node_map_->RemoveOutput(node.name(), consumer->name());
            } else {
              if (IsControlInput(consumer->input(i))) {
                *consumer->mutable_input(i) = AsControlDependency(half_def);
              } else {
                string pos = strings::StrCat(":",
                                             NodePosition(consumer->input(i)));
                *consumer->mutable_input(i) = half_def.name() + pos;
              }
              node_map_->AddOutput(half_def.name(), consumer->name());
              node_map_->RemoveOutput(node.name(), consumer->name());
            }
          }
        }
      }
    } else {
      auto consumers = node_map_->GetOutputs(node.name());
      for (auto consumer : consumers) {
        for (int i = 0; i < consumer->input_size(); ++i) {
          if (NodeName(consumer->input(i)) == node.name()) {
            if (IsControlInput(consumer->input(i))) {
              *consumer->mutable_input(i) = AsControlDependency(half_def);
            } else {
              *consumer->mutable_input(i) = half_def.name();
            }
            node_map_->AddOutput(half_def.name(), consumer->name());
            node_map_->RemoveOutput(node.name(), consumer->name());
          }
        }
      }
    }
    nodes_to_delete.insert(node.name());
    changed_ = true;
  }
  return Status::OK();
}

Status AutoMixedPrecision::HandleBinaryOp(const string& op, const NodeDef& node) {
  if (node.op() == op) {
    std::vector<string> inputs;
    std::vector<string> control_inputs;
    for (auto& input : node.input()) {
      if (IsControlInput(input)) {
        control_inputs.push_back(input);
      } else {
        inputs.push_back(input);
      }
    }
    
    if (node.op() == "Conv2DBackpropInput"
        || node.op() == "Conv2DBackpropFilter") {
      if (inputs.size() != 3) {
        return Status(error::INVALID_ARGUMENT,
                      "Expected 3 inputs for " + node.op());
      }
    } else if (node.op() == "FusedBatchNormGrad") {
      if (inputs.size() != 5) {
        return Status(error::INVALID_ARGUMENT,
                      "Expected 5 inputs for " + node.op());
      }
    } else if (node.op() == "AddN") {
      if (inputs.size() != 2) {
        // only handle AddN with 2 inputs here
        return Status::OK();
      }
    } else {
      if (inputs.size() != 2) {
        return Status(error::INVALID_ARGUMENT,
                      "Expected 2 inputs for " + node.op());
      }
    }

    if (node.op() == "BiasAdd"
        || node.op() == "ReluGrad"
        || node.op() == "AddN"
        || node.op() == "Add") {
      bool from_cast = false;
      for (auto& input : inputs) {
        NodeDef* input_node = node_map_->GetNode(input);
        if (input_node->op() == "Cast" &&
            input_node->attr().at("SrcT").type() == DT_HALF) {
          from_cast = true;
        }
      }
      if (!from_cast) {
        return Status::OK();
      }
    }

    auto cast0_input = inputs[0];
    if (node.op() == "Conv2DBackpropInput") {
      cast0_input = inputs[1];
    }
    node_map_->RemoveOutput(NodeName(cast0_input), node.name());
    NodeDef& cast_def = CreateNodeDefCastf2h(
        /*name=*/node.name() + "/Cast",
        /*device=*/node.device(),
        /*input=*/cast0_input);

    auto cast1_input = inputs[1];
    if (node.op() == "Conv2DBackpropInput"
        || node.op() == "Conv2DBackpropFilter") {
      cast1_input = inputs[2];
    }
    node_map_->RemoveOutput(NodeName(cast1_input), node.name());
    NodeDef& cast1_def = CreateNodeDefCastf2h(
        /*name=*/node.name() + "/Cast_1",
        /*device=*/node.device(),
        /*input=*/cast1_input);

    NodeDef& half_def = *optimized_graph_->add_node();
    auto op = node.op();
    if (op == "FusedBatchNormGrad"){
      op = "FusedBatchNormGradV2";
    }
    half_def.set_op(op);
    half_def.set_name(node.name() + "_amp");
    half_def.set_device(node.device());
    node_map_->AddNode(half_def.name(), &half_def);
    if (node.op() == "Conv2DBackpropInput") {
      half_def.add_input(inputs[0]);
      node_map_->RemoveOutput(NodeName(inputs[0]), node.name());
      node_map_->AddOutput(NodeName(inputs[0]), half_def.name());
    }
    half_def.add_input(cast_def.name());
    node_map_->AddOutput(cast_def.name(), half_def.name());
    if (node.op() == "Conv2DBackpropFilter") {
      half_def.add_input(inputs[1]);
      node_map_->RemoveOutput(NodeName(inputs[1]), node.name());
      node_map_->AddOutput(NodeName(inputs[1]), half_def.name());
    }
    half_def.add_input(cast1_def.name());
    node_map_->AddOutput(cast1_def.name(), half_def.name());
    if (node.op() == "FusedBatchNormGrad") {
      for (int i = 2; i < inputs.size(); ++i) {
        half_def.add_input(inputs[i]);
        node_map_->RemoveOutput(NodeName(inputs[i]), node.name());
        node_map_->AddOutput(NodeName(inputs[i]), half_def.name());
      }
    }
    *half_def.mutable_attr() = node.attr();
    (*half_def.mutable_attr())["T"] = attr_half_;
    if (node.op() == "FusedBatchNormGrad"){
      (*half_def.mutable_attr())["U"] = attr_float_;
    }
    for (auto& control_input : control_inputs) {
      half_def.add_input(control_input);
      node_map_->RemoveOutput(NodeName(control_input), node.name());
      node_map_->AddOutput(NodeName(control_input), half_def.name());
    }

    NodeDef& cast2_def = CreateNodeDefCasth2f(
        /*name=*/node.name() + "/Cast_2",
        /*device=*/node.device(),
        /*input=*/half_def.name());

    auto consumers = node_map_->GetOutputs(node.name());
    for (auto consumer : consumers) {
        for (int i = 0; i < consumer->input_size(); ++i) {
          if (NodeName(consumer->input(i)) == node.name()) {
            if (NodePosition(consumer->input(i)) <= 0) {
              if (IsControlInput(consumer->input(i))) {
                *consumer->mutable_input(i) = AsControlDependency(cast2_def);
              } else {
                *consumer->mutable_input(i) = cast2_def.name();
              }
              node_map_->AddOutput(cast2_def.name(), consumer->name());
              node_map_->RemoveOutput(node.name(), consumer->name());
            } else {
              if (IsControlInput(consumer->input(i))) {
                *consumer->mutable_input(i) = AsControlDependency(half_def);
              } else {
                string pos = strings::StrCat(":",
                                             NodePosition(consumer->input(i)));
                *consumer->mutable_input(i) = half_def.name() + pos;
              }
              node_map_->AddOutput(half_def.name(), consumer->name());
              node_map_->RemoveOutput(node.name(), consumer->name());
            }
          }
        }
    }
    nodes_to_delete.insert(node.name());
    changed_ = true;
  }
  return Status::OK();
}

Status AutoMixedPrecision::RemoveCastPairs() {
  auto nodes = optimized_graph_->node();
  for (const NodeDef& second_cast: nodes) {
    if (second_cast.op() != "Cast") {
      continue;
    }
    std::vector<string> inputs;
    std::vector<string> control_inputs;
    for (auto& input : second_cast.input()) {
      if (IsControlInput(input)) {
        control_inputs.push_back(input);
      } else {
        inputs.push_back(input);
      }
    }
    if (inputs.size() != 1) {
      return Status(error::INVALID_ARGUMENT,
                    "Expected 1 input for " + second_cast.op());
    }
    NodeDef* first_cast = node_map_->GetNode(inputs[0]);
    if (first_cast->op() != "Cast" ||
        first_cast->attr().at("SrcT").type()
        != second_cast.attr().at("DstT").type()) {
      continue;
    }

    inputs.clear();
    for (auto& control_input : control_inputs) {
      node_map_->RemoveOutput(NodeName(control_input), second_cast.name());
    }
    for (auto& input : first_cast->input()) {
      if (IsControlInput(input)) {
        control_inputs.push_back(input);
      } else {
        inputs.push_back(input);
      }
      node_map_->RemoveOutput(NodeName(input), first_cast->name());
    }
    if (inputs.size() != 1) {
      return Status(error::INVALID_ARGUMENT, "Expected 1 input for " + first_cast->op());
    }
    NodeDef* cast_input = node_map_->GetNode(inputs[0]);

    auto consumers = node_map_->GetOutputs(second_cast.name());
    for (auto consumer : consumers) {
        for (int i = 0; i < consumer->input_size(); ++i) {
          if (NodeName(consumer->input(i)) == second_cast.name()) {
            if (IsControlInput(consumer->input(i))) {
              *consumer->mutable_input(i) = AsControlDependency(*cast_input);
            } else {
              *consumer->mutable_input(i) = cast_input->name();
            }
            node_map_->AddOutput(cast_input->name(), consumer->name());
            node_map_->RemoveOutput(second_cast.name(), consumer->name());
            for (auto& control_input : control_inputs) {
              consumer->add_input(control_input);
              node_map_->AddOutput(NodeName(control_input), consumer->name());
            }
          }
        }
    }
    nodes_to_delete.insert(second_cast.name());
    // first_cast may have other consumers, so do not delete
    changed_ = true;
  }
  return Status::OK();
}

Status AutoMixedPrecision::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  optimized_graph_ = optimized_graph;
  /*
  for (const NodeDef& node : optimized_graph_->node()) {
      NodeDef* node_ptr = const_cast<NodeDef*>(&node);
      if (node_ptr->op() == "FusedBatchNorm") {
        node_ptr->set_op("FusedBatchNormV2");
        (*node_ptr->mutable_attr())["U"] = attr_float_;
      }
  }
  */
  node_map_.reset(new NodeMap(optimized_graph_));
  changed_ = true;
  const int num_iterations = 10;
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    if (!changed_) {
      break;
    }
    changed_ = false;
    auto nodes = optimized_graph_->node();
    for (const NodeDef& n : nodes) {
      auto& node = *node_map_->GetNode(n.name());
      if (node.attr().count("T") == 0) {
        continue;
      }

      if (node.attr().at("T").type() != DT_FLOAT) {
        continue;
      }

      if (node.device().find("GPU") == -1) {
        continue;
      }

      TF_RETURN_IF_ERROR(HandleBinaryOp("MatMul", node));
      TF_RETURN_IF_ERROR(HandleBinaryOp("Conv2D", node));
      // Conv2DBackpropFilter is not binary op,
      // but only input0 & input2 need conversion
      TF_RETURN_IF_ERROR(HandleBinaryOp("Conv2DBackpropFilter", node));
      // Conv2DBackpropInput is not binary op,
      // but only input1 & input2 need conversion
      TF_RETURN_IF_ERROR(HandleBinaryOp("Conv2DBackpropInput", node));
      // FusedBatchNormGrad is not binary op,
      // but only input0 & input1 need conversion
      TF_RETURN_IF_ERROR(HandleBinaryOp("FusedBatchNormGrad", node));
      TF_RETURN_IF_ERROR(HandleBinaryOp("BiasAdd", node));
      TF_RETURN_IF_ERROR(HandleBinaryOp("ReluGrad", node));
      TF_RETURN_IF_ERROR(HandleBinaryOp("Add", node));
      TF_RETURN_IF_ERROR(HandleBinaryOp("AddN", node));
      TF_RETURN_IF_ERROR(HandleUnaryOp("Relu", node));
      TF_RETURN_IF_ERROR(HandleUnaryOp("Identity", node));
      TF_RETURN_IF_ERROR(HandleUnaryOp("MaxPool", node));
      TF_RETURN_IF_ERROR(HandleUnaryOp("Shape", node));
      TF_RETURN_IF_ERROR(HandleUnaryOp("BiasAddGrad", node));
      // Reshape is not unary op, but only input0 needs conversion
      TF_RETURN_IF_ERROR(HandleUnaryOp("Reshape", node));
      // FusedBatchNorm is not unary op, but only input0 needs conversion
      TF_RETURN_IF_ERROR(HandleUnaryOp("FusedBatchNorm", node));
    }
    TF_RETURN_IF_ERROR(RemoveCastPairs());
    EraseNodesFromGraph(nodes_to_delete, optimized_graph_);
  }
  return Status::OK();
}

void AutoMixedPrecision::Feedback(Cluster* /*cluster*/, const GrapplerItem& /*item*/,
                             const GraphDef& /*optimized_graph*/,
                             double /*result*/) {
  // Nothing to do for AutoMixedPrecision.
}

}  // end namespace grappler
}  // end namespace tensorflow
