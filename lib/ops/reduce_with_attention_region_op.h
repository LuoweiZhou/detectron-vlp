#ifndef REDUCE_WITH_ATTENTION_REGION_OP_H_
#define REDUCE_WITH_ATTENTION_REGION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ReduceWithAttentionRegionOp final : public Operator<Context> {
 public:
  ReduceWithAttentionRegionOp(const OperatorDef& def, Workspace* ws)
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
class ReduceWithAttentionRegionGradientOp final : public Operator<Context> {
 public:
  ReduceWithAttentionRegionGradientOp(const OperatorDef& def, Workspace* ws)
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

#endif // REDUCE_WITH_ATTENTION_REGION_OP_H_
