#ifndef DIV_CONV_NORM_OP_H_
#define DIV_CONV_NORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class DivConvNormOp final : public Operator<Context> {
 public:
  DivConvNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

template <typename T, class Context>
class DivConvNormGradientOp final : public Operator<Context> {
 public:
  DivConvNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

} // namespace caffe2

#endif // DIV_CONV_NORM_OP_H_
