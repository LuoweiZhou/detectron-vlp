#include "div_conv_norm_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(DivConvNorm, DivConvNormOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(DivConvNormGradient, DivConvNormGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(DivConvNorm)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Division between a feature map and a gate.
)DOC")
    .Input(
        0,
        "X1",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "X2",
        "2D feature map input of shape (N, A, H, W). C is multiple of A.")
    .Output(
        0,
        "Y",
        "4D output of shape (N, C, H, W).");

OPERATOR_SCHEMA(DivConvNormGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Input(
        1,
        "X1",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        2,
        "X2",
        "2D feature map input of shape (N, A, H, W). C is multiple of A.")
    .Output(
        0,
        "dX1",
        "Gradient of forward input 0 (X1).");

class GetDivConvNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "DivConvNormGradient",
        "",
        // Not sure about this, I means input, GO means gradient of the output
        vector<string>{GO(0), I(0), I(1)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(DivConvNorm, GetDivConvNormGradient);

} // namespace caffe2