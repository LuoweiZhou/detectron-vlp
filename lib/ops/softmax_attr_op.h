#ifndef SOFTMAX_ATTR_OP_H_
#define SOFTMAX_ATTR_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class SoftmaxAttrOp final : public Operator<Context> {
 public:
  SoftmaxAttrOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        ignore_(OperatorBase::GetSingleArgument<int>("ignore", -1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int ignore_;
  float scale_;
  Tensor<Context> losses_;
  Tensor<Context> sum_weight_;
  Tensor<Context> weights_;
};

template <typename T, class Context>
class SoftmaxAttrGradientOp final : public Operator<Context> {
 public:
  SoftmaxAttrGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(OperatorBase::GetSingleArgument<float>("scale", 1.)),
        ignore_(OperatorBase::GetSingleArgument<int>("ignore", -1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int ignore_;
  float scale_;
  Tensor<Context> sum_weight_;
  Tensor<Context> weights_;
};

} // namespace caffe2

#endif // SOFTMAX_ATTR_OP_H_