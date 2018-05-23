#include "reduce_boxes_and_feats_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ReduceBoxesAndFeats, ReduceBoxesAndFeatsOp<float, CPUContext>);

OPERATOR_SCHEMA(ReduceBoxesAndFeats)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Reduce the boxes and features from the same image.
)DOC")
    .Arg(
        "im",
        "(int) image index.")
    .Arg(
        "dpi",
        "(int) number of detections per image.")
    .Input(
        0,
        "boxes_in",
        "bounding box information of shape (R, 6): x1, y1, x2, y2, score, c.")
    .Input(
        1,
        "feats_in",
        "2D tensor of shape (R, F) specifying R RoI features.")
    .Output(
        0,
        "rois",
        "2D tensor of shape (R, 5) specifying R RoIs with five columns "
        "representing: batch index in im, x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image.")
    .Output(
        1,
        "feats",
        "2D tensor of shape (R, F) specifying R RoI features.");

SHOULD_NOT_DO_GRADIENT(ReduceBoxesAndFeats);

} // namespace caffe2