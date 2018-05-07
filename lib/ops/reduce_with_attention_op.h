#ifndef REDUCE_WITH_ATTENTION_OP_H_
#define REDUCE_WITH_ATTENTION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ReduceWithAttentionOp final : public Operator<Context> {
 public:
  ReduceWithAttentionOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        iter_(OperatorBase::GetSingleArgument<int>("iter", 2)) {
          DCHECK_GT(iter_, 0);
        }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int iter_;
};

template <typename T, class Context>
class ReduceWithAttentionGradientOp final : public Operator<Context> {
 public:
  ReduceWithAttentionGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        iter_(OperatorBase::GetSingleArgument<int>("iter", 2)) {
          DCHECK_GT(iter_, 0);
        }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int iter_;
};

} // namespace caffe2

#endif // REDUCE_WITH_ATTENTION_OP_H_
