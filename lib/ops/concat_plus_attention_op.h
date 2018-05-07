#ifndef CONCAT_PLUS_ATTENTION_OP_H_
#define CONCAT_PLUS_ATTENTION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ConcatPlusAttentionOp final : public Operator<Context> {
 public:
  ConcatPlusAttentionOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

template <typename T, class Context>
class ConcatPlusAttentionGradientOp final : public Operator<Context> {
 public:
  ConcatPlusAttentionGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

} // namespace caffe2

#endif // CONCAT_PLUS_ATTENTION_OP_H_