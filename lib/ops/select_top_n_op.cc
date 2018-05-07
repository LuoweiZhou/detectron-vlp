#include "select_top_n_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SelectTopN, SelectTopNOp<float, CPUContext>);

OPERATOR_SCHEMA(SelectTopN)
    .NumInputs(1)
    .NumOutputs(2, INT_MAX)
    .SetDoc(R"DOC(
Get the top (potentially top_n) index for each of the feature maps.
)DOC")
    .Arg(
        "top_n",
        "(int) Number of top values to consider for each image.")
    .Input(
        0,
        "X",
        "4D feature map input of shape (N, C, H, W).");

SHOULD_NOT_DO_GRADIENT(SelectTopN);

} // namespace caffe2