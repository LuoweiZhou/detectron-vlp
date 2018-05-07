#ifndef REDUCE_BOXES_ONLY_OP_H_
#define REDUCE_BOXES_ONLY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ReduceBoxesOnlyOp final : public Operator<Context> {
 public:
  ReduceBoxesOnlyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        im_(OperatorBase::GetSingleArgument<int>("im", 0)),
        dpi_(OperatorBase::GetSingleArgument<int>("dpi", 100)),
        k_min_(OperatorBase::GetSingleArgument<int>("k_min", 3)),
        k_max_(OperatorBase::GetSingleArgument<int>("k_max", 7)),
        c_scale_(OperatorBase::GetSingleArgument<int>("c_scale", 224)),
        c_level_(OperatorBase::GetSingleArgument<int>("c_level", 4)) {
    DCHECK_GE(im_, 0);
    DCHECK_GE(dpi_, 1);
    im_float_ = static_cast<float>(im_);
    DCHECK_GE(k_min_, 1);
    DCHECK_GE(k_max_, 1);
    DCHECK_GE(c_scale_, 1);
    DCHECK_GE(c_level_, 1);
    DCHECK_LE(c_level_, 7);
    c_scale_f_ = static_cast<float>(c_scale_);
    c_level_f_ = static_cast<float>(c_level_);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int dpi_;
  int im_;
  float im_float_;
  int k_min_;
  int k_max_;
  int c_scale_;
  int c_level_;
  float c_scale_f_;
  float c_level_f_;

  Tensor<Context> values;
  Tensor<Context> indices;
};

} // namespace caffe2

#endif // REDUCE_BOXES_ONLY_OP_H_
