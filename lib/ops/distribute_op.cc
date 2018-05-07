#include "distribute_op.h"

namespace caffe2 {

namespace {

void _copy(const int size, const float* source, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

}

template<>
bool DistributeOp<float, CPUContext>::RunOnDevice() {
  auto& boxes = Input(0);
  const int num_total = boxes.dim32(0);
  DCHECK_EQ(boxes.dim32(1), 11);
  auto& stats = Input(1);
  const int num_levels = stats.dim32(0);
  DCHECK_EQ(num_levels, OutputSize()-2);
  DCHECK_EQ(num_levels, k_max_ - k_min_ + 1);
  const float* boxes_pointer = boxes.data<float>();
  const int* stats_pointer = stats.data<int>();

  int counters[num_levels];
  int accumulators[num_levels];
  float* rois_pointers[num_levels];
  // first is frois, sorted
  auto* crois = Output(0);
  crois->Resize(num_total, 5);
  float* crois_pointer = crois->mutable_data<float>();
  auto* frois = Output(1);
  frois->Resize(num_total, 6);
  float* frois_pointer = frois->mutable_data<float>();

  accumulators[0] = 0;
  for (int i=0; i<num_levels; i++) {
    counters[i] = 0;
    const int current_lvl_count = stats_pointer[i];
    if (i < num_levels - 1) {
      accumulators[i+1] = accumulators[i] + current_lvl_count;
    }
    auto* rois = Output(i+2);
    rois->Resize(current_lvl_count, 5);
    rois_pointers[i] = rois->mutable_data<float>();
  }

  for (int i=0, bp=0; i<num_total; i++, bp+=11) {
    const int lvl = static_cast<int>(boxes_pointer[bp+5]) - k_min_;
    const int cntc = counters[lvl] * 5;
    const int cntf = cntc + counters[lvl];
    float* rois_pointer = rois_pointers[lvl];
    _copy(5, boxes_pointer + bp, rois_pointer + cntc);

    const int aggc = accumulators[lvl] * 5;
    const int aggf = aggc + accumulators[lvl];
    _copy(5, boxes_pointer + bp, crois_pointer + aggc + cntc);
    frois_pointer[aggf + cntf] = boxes_pointer[bp];
    _copy(5, boxes_pointer + bp + 6, frois_pointer + aggf + cntf + 1);
    counters[lvl] ++;
  }

  return true;
}

REGISTER_CPU_OPERATOR(Distribute, DistributeOp<float, CPUContext>);

OPERATOR_SCHEMA(Distribute)
    .NumInputs(2)
    .NumOutputs(3, INT_MAX)
    .SetDoc(R"DOC(
Distribute the boxes to different levels, and separate them to rois and frois
)DOC")
    .Arg(
        "k_min",
        "(int) The finest level.")
    .Arg(
        "k_max",
        "(int) The coarsest level.")
    .Input(
        0,
        "boxes",
        "bounding box information of shape (R, 11): im, x1, y1, x2, y2, lvl, c, lvl, a, h, w.")
    .Input(
        1,
        "stats",
        "(int) aggregated statistics of shape (num_levels).")
    .Output(
        0,
        "rois",
        "region of interest information for combined levels sorted: im, x1, y1, x2, y2")
    .Output(
        1,
        "frois",
        "feature location information for combined levels sorted: im, c, lvl, a, h, w")
    .Output(
        2,
        "rois_1",
        "region of interest information for level 1: im, x1, y1, x2, y2");


SHOULD_NOT_DO_GRADIENT(Distribute);

} // namespace caffe2