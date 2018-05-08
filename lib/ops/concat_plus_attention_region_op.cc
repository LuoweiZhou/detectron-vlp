#include "concat_plus_attention_region_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ConcatPlusAttentionRegion, ConcatPlusAttentionRegionOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ConcatPlusAttentionRegionGradient, ConcatPlusAttentionRegionGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(ConcatPlusAttentionRegion)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Concatenate and move the axis of the iteration to just before H.
)DOC")
    .Output(
        0,
        "Y",
        "Concatenated tensor.");

OPERATOR_SCHEMA(ConcatPlusAttentionRegionGradient)
    .NumInputs(2, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .Input(
        0,
        "dY",
        "Input gradient for Y.");

class GetConcatPlusAttentionRegionGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // just copy and paste the input to the output
    vector<string> inputs;
    vector<string> grads;
    inputs.push_back(GO(0));
    for (int i = 0; i < def_.input_size(); ++i) {
      inputs.push_back(I(i));
      grads.push_back(GI(i));
    }
    return SingleGradientDef("ConcatPlusAttentionRegionGradient", 
                            "", 
                            inputs, 
                            grads);
  }
};

REGISTER_GRADIENT(ConcatPlusAttentionRegion, GetConcatPlusAttentionRegionGradient);

} // namespace caffe2