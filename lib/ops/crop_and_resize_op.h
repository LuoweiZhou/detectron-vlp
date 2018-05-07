#ifndef CROP_AND_RESIZE_OP_H_
#define CROP_AND_RESIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class CropAndResizeOp final : public Operator<Context> {
 public:
  CropAndResizeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        spatial_scale_(
            OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)),
        pooled_height_(OperatorBase::GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(OperatorBase::GetSingleArgument<int>("pooled_w", 1)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(pooled_height_, 1);
    DCHECK_GT(pooled_width_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
};

template <typename T, class Context>
class CropAndResizeGradientOp final : public Operator<Context> {
 public:
  CropAndResizeGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(
            OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)),
        pooled_height_(OperatorBase::GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(OperatorBase::GetSingleArgument<int>("pooled_w", 1)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(pooled_height_, 1);
    DCHECK_GT(pooled_width_, 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
};

} // namespace caffe2

#endif // CROP_AND_RESIZE_OP_H_