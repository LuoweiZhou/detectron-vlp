#ifndef GET_LOGITS_OP_H_
#define GET_LOGITS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class GetLogitsOp final : public Operator<Context> {
 public:
  GetLogitsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), 
          A_(OperatorBase::GetSingleArgument<int>("A", 9)),
          num_feats_(OperatorBase::GetSingleArgument<int>("num_feats", 80)),
          k_min_(OperatorBase::GetSingleArgument<int>("k_min", 3)),
          k_max_(OperatorBase::GetSingleArgument<int>("k_max", 7)),
          all_dim_(OperatorBase::GetSingleArgument<int>("all_dim", true)) {
        DCHECK_GE(A_, 1);
        DCHECK_GE(num_feats_, 1);
        DCHECK_GE(k_min_, 1);
        DCHECK_GE(k_max_, 1);
        DCHECK_LE(k_min_, k_max_);
      }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int num_feats_;
  int k_min_;
  int k_max_;
  int A_;
  bool all_dim_;
};

} // namespace caffe2

#endif // GET_LOGITS_OP_H_
