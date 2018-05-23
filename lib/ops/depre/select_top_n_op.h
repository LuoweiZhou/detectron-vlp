#ifndef SELECT_TOP_N_OP_H_
#define SELECT_TOP_N_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class SelectTopNOp final : public Operator<Context> {
 public:
  SelectTopNOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        top_n_(OperatorBase::GetSingleArgument<int>("top_n", 1000)) {
    DCHECK_GE(top_n_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int top_n_;
};

} // namespace caffe2

#endif // SELECT_TOP_N_OP_H_
