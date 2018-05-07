#include "class_based_boxes_single_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ClassBasedBoxesSingle, ClassBasedBoxesSingleOp<float, CPUContext>);

OPERATOR_SCHEMA(ClassBasedBoxesSingle)
    .NumInputs(3)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Return class based boxes and features.
)DOC")
    .Arg(
        "class_id",
        "(int) the class to consider in this op, 0-based.")
    .Input(
        0,
        "stats",
        "(int) aggregated statistics of shape (num_cls).")
    .Input(
        1,
        "boxes",
        "bounding box information of shape (R, 7): x1, y1, x2, y2, score, c, valid.")
    .Input(
        2,
        "feats",
        "feature information of shape (R, num_feat).")
    .Output(
        0,
        "boxes_cls",
        "bounding box information for class 1 of shape (T, 8): x1, y1, x2, y2, score, area, c, ZERO.")
    .Output(
        1,
        "feats_cls",
        "feature information for class 1 of shape (T, num_feat).");


SHOULD_NOT_DO_GRADIENT(ClassBasedBoxesSingle);

} // namespace caffe2