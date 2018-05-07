#include "resize_bilinear_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ResizeBilinear, ResizeBilinearOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ResizeBilinearGradient, ResizeBilinearGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(ResizeBilinear)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Resize Bilinear Operation.
)DOC")
    .Arg(
        "height",
        "(int) default 1; Height of the output.")
    .Arg(
        "width",
        "(int) default 1; Width of the output.")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Output(
        0,
        "Y",
        "4D output of shape (N, C, height, width).");

OPERATOR_SCHEMA(ResizeBilinearGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Input(
        0,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Input(
        1,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetResizeBilinearGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeBilinearGradient",
        "",
        // Not sure about this, I means input, GO means gradient of the output
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ResizeBilinear, GetResizeBilinearGradient);

} // namespace caffe2