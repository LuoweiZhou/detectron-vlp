#ifndef INV_CROP_AND_RESIZE_OP_H_
#define INV_CROP_AND_RESIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class InvCropAndResizeOp final : public Operator<Context> {
 public:
  InvCropAndResizeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        spatial_scale_(
            OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)) {
    DCHECK_GT(spatial_scale_, 0);
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
class InvCropAndResizeGradientOp final : public Operator<Context> {
 public:
  InvCropAndResizeGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(
            OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)) {
    DCHECK_GT(spatial_scale_, 0);
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

#endif // INV_CROP_AND_RESIZE_OP_H_