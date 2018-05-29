#ifndef COMPUTE_BOX_TARGETS_OP_H_
#define COMPUTE_BOX_TARGETS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ComputeBoxTargetsOp final : public Operator<Context> {
 public:
  ComputeBoxTargetsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
      num_classes_(OperatorBase::GetSingleArgument<int>("num_classes", 81)) {
        DCHECK_GE(num_classes_, 1);
      }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
 protected:
  int num_classes_;
};

} // namespace caffe2

#endif // COMPUTE_BOX_TARGETS_OP_H_
