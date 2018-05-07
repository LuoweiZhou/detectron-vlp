#ifndef BOXES_AND_FEATS_OP_H_
#define BOXES_AND_FEATS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class BoxesAndFeatsOp final : public Operator<Context> {
 public:
  BoxesAndFeatsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        A_(OperatorBase::GetSingleArgument<int>("A", 9)),
        im_(OperatorBase::GetSingleArgument<int>("im", 0)),
        lvl_(OperatorBase::GetSingleArgument<int>("level", 3)) {
    DCHECK_GE(A_, 1);
    DCHECK_GE(im_, 0);
    DCHECK_GE(lvl_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int A_;
  int im_;
  int lvl_;
};

} // namespace caffe2

#endif // BOXES_AND_FEATS_OP_H_
