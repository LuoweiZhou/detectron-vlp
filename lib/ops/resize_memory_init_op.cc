#include "resize_memory_init_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ResizeMemoryInit, ResizeMemoryInitOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ResizeMemoryInitGradient, ResizeMemoryInitGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(ResizeMemoryInit)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Resize Memory operation for initialization.
)DOC")
    .Arg(
        "spatial_scale",
        "(float) default 1.; Scale of the memory.")
    .Arg(
        "e_value",
        "(float) default 0.; extrapolation value.")
    .Input(
        0,
        "X",
        "3D feature map input of shape (C, H, W).")
    .Input(
        1,
        "image_info",
        "Image information.")
    .Input(
        2,
        "reference",
        "Reference blob to get the height, width, and number of images.")
    .Output(
        0,
        "Y",
        "4D output of shape (N, C, height, width).");

OPERATOR_SCHEMA(ResizeMemoryInitGradient)
    .NumInputs(4)
    .NumOutputs(1)
    .Input(
        0,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Input(
        1,
        "X",
        "3D feature map input of shape (C, H, W).")
    .Input(
        2,
        "image_info",
        "Image information.")
    .Input(
        3,
        "reference",
        "Reference blob to get the height, width, and number of images.")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetResizeMemoryInitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeMemoryInitGradient",
        "",
        // Not sure about this, I means input, GO means gradient of the output
        vector<string>{GO(0), I(0), I(1), I(2)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ResizeMemoryInit, GetResizeMemoryInitGradient);

} // namespace caffe2