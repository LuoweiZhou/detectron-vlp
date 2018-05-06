#include "softmax_attr_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SoftmaxAttr, SoftmaxAttrOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SoftmaxAttrGradient,
    SoftmaxAttrGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SoftmaxAttr)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Peter Anderson's code to deal with attributes.
)DOC")
    .Arg(
        "scale",
        "(float) default 1.0; multiply the loss by this scale factor.")
    .Arg(
        "ignore",
        "(int) default -1, class label to ignore in each row.")
    .Input(
        0,
        "scores",
        "2D tensor of softmax inputs, of shape (R, C).")
    .Input(
        1,
        "labels",
        "2D tensor of labels, of shape (R, G), where G is the number of attributes in each group.")
    .Output(
        0,
        "prob",
        "probabilities.")
    .Output(
        1,
        "loss",
        "Scalar loss.");

OPERATOR_SCHEMA(SoftmaxAttrGradient)
    .NumInputs(4)
    .NumOutputs(1)
    .Input(
        0,
        "scores",
        "See SoftmaxAttr.")
    .Input(
        1,
        "labels",
        "See SoftmaxAttr.")
    .Input(
        2,
        "prob",
        "Output 0 from SoftmaxAttr; See SoftmaxAttr.")
    .Input(
        3,
        "d_loss",
        "Gradient of forward output 0 (loss)")
    .Output(
        0,
        "d_scores",
        "Gradient of forward input 0 (scores)");

class GetSoftmaxAttrGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SoftmaxAttrGradient",
        "",
        vector<string>{I(0), I(1), O(0), GO(1)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SoftmaxAttr, GetSoftmaxAttrGradient);
} // namespace caffe2