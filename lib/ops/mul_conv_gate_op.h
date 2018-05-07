#ifndef MUL_CONV_GATE_OP_H_
#define MUL_CONV_GATE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class MulConvGateOp final : public Operator<Context> {
 public:
  MulConvGateOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

template <typename T, class Context>
class MulConvGateGradientOp final : public Operator<Context> {
 public:
  MulConvGateGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

} // namespace caffe2

#endif // MUL_CONV_GATE_OP_H_
