#ifndef RESIZE_BILINEAR_AS_OP_H_
#define RESIZE_BILINEAR_AS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ResizeBilinearAsOp final : public Operator<Context> {
 public:
  ResizeBilinearAsOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)) {
    DCHECK_GE(spatial_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
};

template <typename T, class Context>
class ResizeBilinearAsGradientOp final : public Operator<Context> {
 public:
  ResizeBilinearAsGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)) {
    DCHECK_GE(spatial_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
};

} // namespace caffe2

#endif // RESIZE_BILINEAR_AS_OP_H_
