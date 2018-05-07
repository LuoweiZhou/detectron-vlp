#ifndef DISTRIBUTE_OP_H_
#define DISTRIBUTE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class DistributeOp final : public Operator<Context> {
 public:
  DistributeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
      k_min_(OperatorBase::GetSingleArgument<int>("k_min", 3)),
      k_max_(OperatorBase::GetSingleArgument<int>("k_max", 7)) {
        DCHECK_GE(k_min_, 1);
        DCHECK_GE(k_max_, 1);
      }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int k_min_;
  int k_max_;
};

} // namespace caffe2

#endif // DISTRIBUTE_OP_H_
