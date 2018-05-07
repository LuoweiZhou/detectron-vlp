#include "boxes_and_feats_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(BoxesAndFeats, BoxesAndFeatsOp<float, CPUContext>);

OPERATOR_SCHEMA(BoxesAndFeats)
    .NumInputs(6)
    .NumOutputs(3)
    .SetDoc(R"DOC(
Return boxes and features given indexes.
)DOC")
    .Arg(
        "A",
        "(int) Product of scale and aspect ratio per location.")
    .Arg(
        "im",
        "(int) Image Index.")
    .Arg(
        "level",
        "(int) Level of the current detection.")
    .Input(
        0,
        "cls_preds",
        "4D feature map input of shape (N, A * num_cls, H, W).")
    .Input(
        1,
        "box_preds",
        "4D feature map input of shape (N, A * 4, H, W).")
    .Input(
        2,
        "anchors",
        "2D feature map input of shape (A, 4).")
    .Input(
        3,
        "YI",
        "location index of shape (R).")
    .Input(
        4,
        "YV",
        "location value of shape (R).")
    .Input(
        5,
        "im_info",
        "image information.")
    .Output(
        0,
        "boxes",
        "bounding box information of shape (R, 7): x1, y1, x2, y2, score, c, valid.")
    .Output(
        1,
        "feats",
        "feature information of shape (R, num_cls).")
    .Output(
        2,
        "stats",
        "Number of bounding boxes assigned to each class. (num_cls)");


SHOULD_NOT_DO_GRADIENT(BoxesAndFeats);

} // namespace caffe2