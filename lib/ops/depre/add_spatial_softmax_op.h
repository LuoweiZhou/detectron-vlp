#ifndef ADD_SPATIAL_SOFTMAX_OP_H_
#define ADD_SPATIAL_SOFTMAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class AddSpatialSoftmaxOp final : public Operator<Context> {
 public:
  AddSpatialSoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_classes_(OperatorBase::GetSingleArgument<int>("num_classes", 2)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int num_classes_;
  StorageOrder order_;
};

template <typename T, class Context>
class AddSpatialSoftmaxGradientOp final : public Operator<Context> {
 public:
  AddSpatialSoftmaxGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        num_classes_(OperatorBase::GetSingleArgument<int>("num_classes", 2)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  int num_classes_;
  StorageOrder order_;
  Tensor<Context> sum_probs_;
};

} // namespace caffe2

#endif // ADD_SPATIAL_SOFTMAX_OP_H_