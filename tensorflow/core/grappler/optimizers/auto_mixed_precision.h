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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_

#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

namespace tensorflow {
namespace grappler {
class AutoMixedPrecision : public GraphOptimizer {
 public:
  AutoMixedPrecision() {
    attr_float_.set_type(DT_FLOAT);
    attr_half_.set_type(DT_HALF);
    attr_false_.set_b(false);
  }
  ~AutoMixedPrecision() override {}

  string name() const override { return "amp"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 private:
  Status HandleBinaryOp(const string& op, const NodeDef& node);
  Status HandleUnaryOp(const string& op, const NodeDef& node);
  Status RemoveCastPairs();
  NodeDef& CreateNodeDefCasth2f(const string& name, const string& device, const string& input);
  NodeDef& CreateNodeDefCastf2h(const string& name, const string& device, const string& input);

  std::unique_ptr<NodeMap> node_map_;
  GraphDef* optimized_graph_;  // Not owned.

  AttrValue attr_float_;
  AttrValue attr_half_;
  AttrValue attr_false_;

  bool changed_;
  std::set<string> nodes_to_delete;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_
