#ifndef RESIZE_MEMORY_INIT_OP_H_
#define RESIZE_MEMORY_INIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ResizeMemoryInitOp final : public Operator<Context> {
 public:
  ResizeMemoryInitOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)),
        e_value_(OperatorBase::GetSingleArgument<float>("e_value", 0.)) {
          DCHECK_GT(spatial_scale_, 0);
        }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  float e_value_;
};

template <typename T, class Context>
class ResizeMemoryInitGradientOp final : public Operator<Context> {
 public:
  ResizeMemoryInitGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)),
        e_value_(OperatorBase::GetSingleArgument<float>("e_value", 0.)) {
          DCHECK_GT(spatial_scale_, 0);
        }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  float e_value_;
};

} // namespace caffe2

#endif // RESIZE_MEMORY_INIT_OP_H_
