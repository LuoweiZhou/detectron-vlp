#ifndef SUM_CONV_FC_OP_H_
#define SUM_CONV_FC_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class SumConvFCOp final : public Operator<Context> {
 public:
  SumConvFCOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

template <typename T, class Context>
class SumConvFCGradientOp final : public Operator<Context> {
 public:
  SumConvFCGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

} // namespace caffe2

#endif // SUM_CONV_FC_OP_H_
