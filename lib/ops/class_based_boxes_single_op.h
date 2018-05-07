#ifndef CLASS_BASED_BOXES_OP_H_
#define CLASS_BASED_BOXES_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ClassBasedBoxesSingleOp final : public Operator<Context> {
 public:
  ClassBasedBoxesSingleOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
      class_(OperatorBase::GetSingleArgument<int>("class_id", 0)) {
        class_float_ = static_cast<float>(class_);
        counter_.Resize(1);
      }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int class_;
  float class_float_;
  Tensor<Context> counter_;
  int* counter_pointer_;
};

} // namespace caffe2

#endif // CLASS_BASED_BOXES_OP_H_
