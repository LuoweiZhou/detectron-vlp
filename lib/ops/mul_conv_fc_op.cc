#include "mul_conv_fc_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(MulConvFC, MulConvFCOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MulConvFCGradient, MulConvFCGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(MulConvFC)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Multiplication between a convolution feature and a fully connected feature.
)DOC")
    .Input(
        0,
        "X1",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "X2",
        "2D feature map input of shape (N, C).")
    .Output(
        0,
        "Y",
        "4D output of shape (N, C, H, W).");

OPERATOR_SCHEMA(MulConvFCGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .Input(
        0,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Input(
        1,
        "X1",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "X2",
        "2D feature map input of shape (N, C).")
    .Output(
        0,
        "dX1",
        "Gradient of forward input 0 (X1).")
    .Output(
        1,
        "dX2",
        "Gradient of forward input 1 (X2).");

class GetMulConvFCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MulConvFCGradient",
        "",
        // Not sure about this, I means input, GO means gradient of the output
        vector<string>{GO(0), I(0), I(1)},
        vector<string>{GI(0), GI(1)});
  }
};

REGISTER_GRADIENT(MulConvFC, GetMulConvFCGradient);

} // namespace caffe2