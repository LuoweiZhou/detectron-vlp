#include "class_based_boxes_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ClassBasedBoxes, ClassBasedBoxesOp<float, CPUContext>);

OPERATOR_SCHEMA(ClassBasedBoxes)
    .NumInputs(3)
    .NumOutputs(2, INT_MAX)
    .SetDoc(R"DOC(
Return class based boxes and features.
)DOC")
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
        "boxes1",
        "bounding box information for class 1 of shape (T, 8): x1, y1, x2, y2, score, area, c, ZERO.")
    .Output(
        1,
        "feats1",
        "feature information for class 1 of shape (T, num_feat).");


SHOULD_NOT_DO_GRADIENT(ClassBasedBoxes);

} // namespace caffe2