#include "embed_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Embed, EmbedOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(EmbedGradient, EmbedGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Embed)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Sum between a convolution feature and a fully connected feature.
)DOC")
    .Input(
        0,
        "X",
        "2D feature map input of shape (N, C).")
    .Input(
        1,
        "I",
        "1D index of shape (I).")
    .Output(
        0,
        "Y",
        "2D feature map input of shape (I, C).");

OPERATOR_SCHEMA(EmbedGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Input(
        1,
        "X",
        "2D feature map input of shape (N, C).")
    .Input(
        2,
        "I",
        "1D index of shape (I).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetEmbedGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "EmbedGradient",
        "",
        // Not sure about this, I means input, GO means gradient of the output
        vector<string>{GO(0), I(0), I(1)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Embed, GetEmbedGradient);

} // namespace caffe2