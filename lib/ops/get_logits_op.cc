#include "get_logits_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(GetLogits, GetLogitsOp<float, CPUContext>);

OPERATOR_SCHEMA(GetLogits)
    .NumInputs(2, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Get the features given the locations.
)DOC")
    .Arg(
        "num_feats",
        "Number of features in each row.")
    .Arg(
        "k_min",
        "The bottom most feature pyramid layer.")
    .Arg(
        "k_max",
        "The top most feature pyramid layer.")
    .Arg(
        "all_dim",
        "If copy for all the dimensions.")
    .Input(
        0,
        "frois",
        "(float) tensor of shape (R, 6), each column being (im, c, lvl, a, h, w).")
    .Input(
        1,
        "feats1",
        "(float) prediction features for a class.")
    .Output(
        0,
        "feats_out",
        "(float) output features of shape (R, num_feat).");

SHOULD_NOT_DO_GRADIENT(GetLogits);

} // namespace caffe2