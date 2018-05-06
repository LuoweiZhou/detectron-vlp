#ifndef EMBED_OP_H_
#define EMBED_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class EmbedOp final : public Operator<Context> {
 public:
  EmbedOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

template <typename T, class Context>
class EmbedGradientOp final : public Operator<Context> {
 public:
  EmbedGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }
};

} // namespace caffe2

#endif // EMBED_OP_H_
