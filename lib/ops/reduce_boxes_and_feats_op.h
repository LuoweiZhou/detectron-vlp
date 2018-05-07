#ifndef REDUCE_BOXES_AND_FEATS_OP_H_
#define REDUCE_BOXES_AND_FEATS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ReduceBoxesAndFeatsOp final : public Operator<Context> {
 public:
  ReduceBoxesAndFeatsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        im_(OperatorBase::GetSingleArgument<int>("im", 0)),
        dpi_(OperatorBase::GetSingleArgument<int>("dpi", 100)) {
    DCHECK_GE(im_, 0);
    DCHECK_GE(dpi_, 1);
    im_float_ = static_cast<float>(im_);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int dpi_;
  int im_;
  float im_float_;

  Tensor<Context> values;
  Tensor<Context> indices;
};

} // namespace caffe2

#endif // REDUCE_BOXES_AND_FEATS_OP_H_
