#include "resize_bilinear_as_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ResizeBilinearAs, ResizeBilinearAsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ResizeBilinearAsGradient, ResizeBilinearAsGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(ResizeBilinearAs)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Resize Bilinear as the size of the other input.
)DOC")
    .Arg(
        "spatial_scale",
        "(float) default 1.;")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "R",
        "4D feature map input of shape (N, C, H2, W2).")
    .Output(
        0,
        "Y",
        "4D output of shape (N, C, H2, W2).");

OPERATOR_SCHEMA(ResizeBilinearAsGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Input(
        1,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        2,
        "R",
        "4D feature map input of shape (N, C, H2, W2).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetResizeBilinearAsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeBilinearAsGradient",
        "",
        // Not sure about this, I means input, GO means gradient of the output
        vector<string>{GO(0), I(0), I(1)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ResizeBilinearAs, GetResizeBilinearAsGradient);

} // namespace caffe2