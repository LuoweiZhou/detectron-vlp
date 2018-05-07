#include "crop_and_resize_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(CropAndResize, CropAndResizeOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(CropAndResizeGradient, CropAndResizeGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(CropAndResize)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Crop and resize Operation.
)DOC")
    .Arg(
        "spatial_scale",
        "(float) default 1.0; Spatial scale of the input feature map X "
        "relative to the input image. E.g., 0.0625 if X has a stride of 16 "
        "w.r.t. the input image.")
    .Arg(
        "pooled_h",
        "(int) default 1; Pooled output Y's height.")
    .Arg(
        "pooled_w",
        "(int) default 1; Pooled output Y's width.")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Output(
        0,
        "Y",
        "4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element "
        "is a pooled feature map corresponding to the r-th RoI.");

OPERATOR_SCHEMA(CropAndResizeGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Input(
        2,
        "dY",
        "Gradient of forward output 0 (Y)")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X)");

class GetCropAndResizeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CropAndResizeGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(CropAndResize, GetCropAndResizeGradient);

} // namespace caffe2