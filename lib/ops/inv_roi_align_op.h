#ifndef INV_ROI_ALIGN_OP_H_
#define INV_ROI_ALIGN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class InvRoIAlignOp final : public Operator<Context> {
 public:
  InvRoIAlignOp(const OperatorDef& operator_def, Workspace* ws)
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
class InvRoIAlignGradientOp final : public Operator<Context> {
 public:
  InvRoIAlignGradientOp(const OperatorDef& def, Workspace* ws)
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

#endif // INV_ROI_ALIGN_OP_H_