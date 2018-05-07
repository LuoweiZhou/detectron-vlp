#ifndef CLASS_BASED_BOXES_OP_H_
#define CLASS_BASED_BOXES_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ClassBasedBoxesCombinedOp final : public Operator<Context> {
 public:
  ClassBasedBoxesCombinedOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    size_ratio_ = sizeof(float*) / sizeof(long);
    DCHECK_EQ(size_ratio_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  Tensor<Context> counter_;
  Tensor<Context> pointer_boxes_;
  int size_ratio_;
};

} // namespace caffe2

#endif // CLASS_BASED_BOXES_OP_H_
