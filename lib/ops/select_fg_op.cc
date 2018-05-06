#include "select_fg_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SelectFG, SelectFGOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SelectFGGradient, SelectFGGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SelectFG)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Select the foreground features.
)DOC")
    .Input(
        0,
        "X",
        "2D feature map input of shape (N, C).")
    .Input(
        1,
        "I",
        "1D indicator of shape (I).")
    .Output(
        0,
        "Y",
        "2D feature map input of shape (F, C).");

OPERATOR_SCHEMA(SelectFGGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "dY",
        "Gradient of forward output 0 (Y).")
    .Input(
        1,
        "X",
        "2D feature map input of shape (N, C).")
    .Input(
        2,
        "I",
        "1D indicator of shape (I).")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X).");

class GetSelectFGGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SelectFGGradient",
        "",
        // Not sure about this, I means input, GO means gradient of the output
        vector<string>{GO(0), I(0), I(1)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(SelectFG, GetSelectFGGradient);

} // namespace caffe2