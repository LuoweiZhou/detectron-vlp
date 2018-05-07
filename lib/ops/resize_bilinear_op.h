#ifndef RESIZE_BILINEAR_OP_H_
#define RESIZE_BILINEAR_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ResizeBilinearOp final : public Operator<Context> {
 public:
  ResizeBilinearOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        height_(OperatorBase::GetSingleArgument<int>("height", 1)),
        width_(OperatorBase::GetSingleArgument<int>("width", 1)) {
    DCHECK_GE(height_, 1);
    DCHECK_GE(width_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int height_;
  int width_;
};

template <typename T, class Context>
class ResizeBilinearGradientOp final : public Operator<Context> {
 public:
  ResizeBilinearGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        height_(OperatorBase::GetSingleArgument<int>("height", 1)),
        width_(OperatorBase::GetSingleArgument<int>("width", 1)) {
    DCHECK_GE(height_, 1);
    DCHECK_GE(width_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int height_;
  int width_;
};

} // namespace caffe2

#endif // RESIZE_BILINEAR_OP_H_
