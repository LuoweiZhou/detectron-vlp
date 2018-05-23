#include "add_spatial_softmax_op.h"
#include "caffe2/operators/softmax_shared.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    AddSpatialSoftmax,
    AddSpatialSoftmaxOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    AddSpatialSoftmaxGradient,
    AddSpatialSoftmaxGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(AddSpatialSoftmax)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
A specific form of spatial softmax. Assume the input ignores the implicit
baseline classifier weights.
)DOC")
    .Arg(
        "num_classes",
        "(int) default 2; number of classes in each softmax group.")
    .Input(
        0,
        "scores",
        "4D tensor of softmax inputs (called 'scores' or 'logits') with shape "
        "(N, C, H, W), where C = num_anchors * (num_classes - 1) defines num_anchors "
        "groups of contiguous num_classes softmax inputs.")
    .Output(
        0,
        "probabilities",
        "4D tensor of softmax probabilities with shape (N, C, H, W), where "
        "C = num_anchors * num_classes, and softmax was applied to each of the "
        "num_anchors groups; within a group the num_classes values sum to 1.");

OPERATOR_SCHEMA(AddSpatialSoftmaxGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Input(
        0,
        "scores",
        "See AddSpatialSoftmax")
    .Input(
        1,
        "d_probabilities",
        "Gradient of forward output 0 (probabilities).")
    .Output(
        0,
        "d_scores",
        "Gradient of forward input 0 (scores).");

class GetAddSpatialSoftmaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AddSpatialSoftmaxGradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(AddSpatialSoftmax, GetAddSpatialSoftmaxGradient);
} // namespace caffe2