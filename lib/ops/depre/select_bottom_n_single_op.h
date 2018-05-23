#ifndef SELECT_BOTTOM_N_SINGLE_OP_H_
#define SELECT_BOTTOM_N_SINGLE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class SelectBottomNSingleOp final : public Operator<Context> {
 public:
  SelectBottomNSingleOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        im_(OperatorBase::GetSingleArgument<int>("im", 0)),
        top_n_(OperatorBase::GetSingleArgument<int>("top_n", 1000)) {
    DCHECK_GE(im_, 0);
    DCHECK_GE(top_n_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int im_;
  int top_n_;
};

} // namespace caffe2

#endif // SELECT_BOTTOM_N_SINGLE_OP_H_
