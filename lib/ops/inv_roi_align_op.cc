#include "inv_roi_align_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(InvRoIAlign, InvRoIAlignOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(InvRoIAlignGradient, InvRoIAlignGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(InvRoIAlign)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Inverse of RoIAlign Operation.
)DOC")
    .Arg(
        "spatial_scale",
        "(float) default 1.0; Spatial scale of the input feature map X "
        "relative to the input image. E.g., 0.0625 if X has a stride of 16 "
        "w.r.t. the input image.")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W). Here we only need the "
        "size.")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Input(
        2,
        "RX",
        "4D input of shape (R, C, pooled_h, pooled_w). The r-th batch element "
        "is a feature map corresponding to the r-th RoI.")
    .Output(
        0,
        "Y",
        "4D feature map output of shape (N, C, H, W).");

OPERATOR_SCHEMA(InvRoIAlignGradient)
    .NumInputs(4)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W). Here we only need the "
        "size.")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Input(
        2,
        "RX",
        "4D input of shape (R, C, pooled_h, pooled_w). The r-th batch element "
        "is a feature map corresponding to the r-th RoI.")
    .Input(
        3,
        "dY",
        "Gradient of forward output 0 (Y)")
    .Output(
        0,
        "dRX",
        "Gradient of forward input 2 (RX)");

class GetInvRoIAlignGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "InvRoIAlignGradient",
        "",
        vector<string>{I(0), I(1), I(2), GO(0)},
        vector<string>{GI(2)});
  }
};

REGISTER_GRADIENT(InvRoIAlign, GetInvRoIAlignGradient);

} // namespace caffe2