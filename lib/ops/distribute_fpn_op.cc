#include "distribute_fpn_op.h"

using std::vector;

namespace caffe2 {

namespace {

float _assign_level(const float area, const int k_min, const int k_max, 
                    const float s0, const float k) {
  const float s = sqrt(area);
  float lvl = floor(log2(s / s0 + 1e-6)) + k;
  lvl = (lvl < k_min) ? k_min : lvl;
  lvl = (lvl > k_max) ? k_max : lvl;
  return lvl;
}

void _copy(const int size, const float* source, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

void _zeros(const int size, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = 0.;
    }
}

void _zeros(const int size, int* target) {
    for (int i=0; i<size; i++) {
        *(target++) = 0;
    }
}

}

template<>
bool DistributeFPNOp<float, CPUContext>::RunOnDevice() {
  auto& rois = Input(0);
  const int R = rois.dim32(0);
  const float* rois_pointer = rois.data<float>();
  auto* rois_idx_restore = Output(0);
  rois_idx_restore->Resize(R);
  int* rois_idx_pointer = rois_idx_restore->mutable_data<int>();
  DCHECK_EQ(num_levels_, OutputSize() - 1);
  // first get the statistics
  int* stats_pointer = stats_.mutable_data<int>();
  int* counters_pointer = counters_.mutable_data<int>();
  int* accs_pointer = accs_.mutable_data<int>();
  _zeros(num_levels_, stats_pointer);
  _zeros(num_levels_, counters_pointer);
  levels_.Resize(R);
  int* levels_pointer = levels_.mutable_data<int>();
  for (int i=0; i<R; i++) {
    const int Bp = i * 5;
    const float area = (rois_pointer[Bp + 3] - rois_pointer[Bp + 1] + 1.) 
                      * (rois_pointer[Bp + 4] - rois_pointer[Bp + 2] + 1.);
    const int lvl = static_cast<int>(_assign_level(area, 
                                                  k_min_, 
                                                  k_max_, 
                                                  c_scale_f_, 
                                                  c_level_f_)) - k_min_;
    levels_pointer[i] = lvl;
    stats_pointer[lvl]++;
  }
  int now = 0;
  float* fpn_rois_pointers[num_levels_];
  for (int i=0; i<num_levels_; i++) {
    accs_pointer[i] = now;
    now += stats_pointer[i];
    // also need to initialize the size of the outputs
    auto* rois_fpn = Output(i+1);
    rois_fpn->Resize(stats_pointer[i], 5);
    fpn_rois_pointers[i] = rois_fpn->mutable_data<float>();
  }
  
  DCHECK_EQ(now, R);
  // then outputs the rois, from lowest level up
  for (int i=0; i<R; i++) {
    const int Bp = i * 5;
    const int lvl = levels_pointer[i];
    const int cnt = counters_pointer[lvl]++;
    const int Rp = cnt * 5;
    _copy(5, rois_pointer + Bp, fpn_rois_pointers[lvl] + Rp);
    rois_idx_pointer[i] = accs_pointer[lvl] + cnt;
  }

  return true;
}

REGISTER_CPU_OPERATOR(DistributeFPN, DistributeFPNOp<float, CPUContext>);

OPERATOR_SCHEMA(DistributeFPN)
    .NumInputs(1)
    .NumOutputs(2, INT_MAX)
    .SetDoc(R"DOC(
DistributeFPN the boxes to different levels, and separate them to rois and frois
)DOC")
    .Arg(
        "k_min",
        "(int) The finest level.")
    .Arg(
        "k_max",
        "(int) The coarsest level.")
    .Input(
        0,
        "rois",
        "bounding box information of shape (R, 5): im, x1, y1, x2, y2.")
    .Output(
        0,
        "rois_idx_restore",
        "region of interest index to restore.")
    .Output(
        1,
        "rois_1",
        "region of interest information for level 1 (R1, 5): im, x1, y1, x2, y2.")
    ;


SHOULD_NOT_DO_GRADIENT(DistributeFPN);

} // namespace caffe2