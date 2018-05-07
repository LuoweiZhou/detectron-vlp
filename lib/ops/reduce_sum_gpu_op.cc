#include "reduce_sum_gpu_op.h"

namespace caffe2 {

namespace {

void _copy(const int size, const int* source, int* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

void _add(const int size, const int* source, int* target) {
    for (int i=0; i<size; i++) {
        *(target++) += *(source++);
    }
}

}

template<>
bool ReduceSumGPUOp<int, CPUContext>::RunOnDevice() {
  const int num_inputs = InputSize();  
  auto* Y = Output(0);
  int num_feat;
  int* stats;

  for (int i=0; i<num_inputs; i++) {
    auto& X = Input(i);
    const int* Xp = X.data<int>();

    if (i==0) {
      num_feat = X.dim32(0);
      Y->Resize(num_feat);
      stats = Y->mutable_data<int>();
      _copy(num_feat, Xp, stats);
    } else {
      _add(num_feat, Xp, stats);
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(ReduceSumGPU, ReduceSumGPUOp<int, CPUContext>);

OPERATOR_SCHEMA(ReduceSumGPU)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Reduce by summing up the values in statistics.
)DOC")
    .Input(
        0,
        "stats1",
        "(int) statistics from one layer.")
    .Output(
        0,
        "stats_total",
        "(int) total stats after aggregation.");

SHOULD_NOT_DO_GRADIENT(ReduceSumGPU);

} // namespace caffe2