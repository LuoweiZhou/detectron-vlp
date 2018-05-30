#include "reduce_boxes_only_op.h"

using std::pair;
using std::make_pair;
using std::priority_queue;

namespace caffe2 {

namespace {

// to compare two values
template <typename T>
struct _compare_value {
  bool operator()(
      const pair<T, int>& lhs,
      const pair<T, int>& rhs) {
    return (lhs.first > rhs.first);
  }
};

int _build_heap(priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>>* PQ,
                const float* boxes_pointer, const int num_total, const int top_n) {
    for (int i=0; i<num_total; i++) {
        const float prob = boxes_pointer[i * 12 + 4];
        if (PQ->size() < top_n || prob > PQ->top().first) {
            PQ->push(make_pair(prob, i));
        }
        if (PQ->size() > top_n) {
            PQ->pop();
        }
    }
    return PQ->size();
}

float _assign_level(const float area, const int k_min, const int k_max, const float s0, const float k) {
  const float s = sqrt(area);
  float lvl = floor(log2(s / s0 + 1e-6)) + k;
  lvl = (lvl < k_min) ? k_min : lvl;
  lvl = (lvl > k_max) ? k_max : lvl;
  return lvl;
}

void _zeros(const int size, int* target) {
    for (int i=0; i<size; i++) {
        *(target++) = 0;
    }
}

}

template<>
bool ReduceBoxesOnlyOp<float, CPUContext>::RunOnDevice() {
  auto& boxes = Input(0);
  const int num_total = boxes.dim32(0);
  const int num_levels = k_max_ - k_min_ + 1;
  const float* boxes_pointer = boxes.data<float>();

  if (num_total == 0) {
    Output(0)->Resize(0, 11);
    Output(1)->Resize(num_levels);
    Output(0)->mutable_data<float>();
    int* output_stats_pointer = Output(1)->mutable_data<int>();
    // set the stats to zero
    _zeros(num_levels, output_stats_pointer);
    return true;
  } else if (num_total <= dpi_) {
    auto* output_boxes = Output(0);
    output_boxes->Resize(num_total, 11);
    float* output_boxes_pointer = output_boxes->mutable_data<float>();

    auto* output_stats = Output(1);
    output_stats->Resize(num_levels);
    int* output_stats_pointer = output_stats->mutable_data<int>();
    _zeros(num_levels, output_stats_pointer);

    // copy directly
    for (int i=0, j=0, k=0; i<num_total; i++, j+=12, k+=11) {
        output_boxes_pointer[k] = im_float_;
        output_boxes_pointer[k+1] = boxes_pointer[j];
        output_boxes_pointer[k+2] = boxes_pointer[j+1];
        output_boxes_pointer[k+3] = boxes_pointer[j+2];
        output_boxes_pointer[k+4] = boxes_pointer[j+3];
        const float lvl_assign = _assign_level(boxes_pointer[j+5], 
                                                k_min_, 
                                                k_max_, 
                                                c_scale_f_, 
                                                c_level_f_);
        output_boxes_pointer[k+5] = lvl_assign;
        const int lvl = static_cast<int>(lvl_assign) - k_min_;
        output_stats_pointer[lvl] ++;

        output_boxes_pointer[k+6] = boxes_pointer[j+6];
        output_boxes_pointer[k+7] = boxes_pointer[j+8];
        output_boxes_pointer[k+8] = boxes_pointer[j+9];
        output_boxes_pointer[k+9] = boxes_pointer[j+10];
        output_boxes_pointer[k+10] = boxes_pointer[j+11];
    }
    return true;
  }

  // get scores
  priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>> PQ;
  int num_left = _build_heap(&PQ, boxes_pointer, num_total, dpi_);

  // final outputs
  auto* output_boxes = Output(0);
  output_boxes->Resize(num_left, 11);
  float* output_boxes_pointer = output_boxes->mutable_data<float>();

  auto* output_stats = Output(1);
  output_stats->Resize(num_levels);
  int* output_stats_pointer = output_stats->mutable_data<int>();
  _zeros(num_levels, output_stats_pointer);
  
  for (int i=0, k=0; i<num_left; i++, k+=11) {
    auto& pqelm = PQ.top();
    const int j = (pqelm.second) * 12;
    output_boxes_pointer[k] = im_float_;
    output_boxes_pointer[k+1] = boxes_pointer[j];
    output_boxes_pointer[k+2] = boxes_pointer[j+1];
    output_boxes_pointer[k+3] = boxes_pointer[j+2];
    output_boxes_pointer[k+4] = boxes_pointer[j+3];
    const float lvl_assign = _assign_level(boxes_pointer[j+5], 
                                            k_min_, 
                                            k_max_, 
                                            c_scale_f_, 
                                            c_level_f_);
    output_boxes_pointer[k+5] = lvl_assign;
    const int lvl = static_cast<int>(lvl_assign) - k_min_;
    output_stats_pointer[lvl] ++;

    output_boxes_pointer[k+6] = boxes_pointer[j+6];
    output_boxes_pointer[k+7] = boxes_pointer[j+8];
    output_boxes_pointer[k+8] = boxes_pointer[j+9];
    output_boxes_pointer[k+9] = boxes_pointer[j+10];
    output_boxes_pointer[k+10] = boxes_pointer[j+11];
    PQ.pop();
  }

  return true;
}

REGISTER_CPU_OPERATOR(ReduceBoxesOnly, ReduceBoxesOnlyOp<float, CPUContext>);

OPERATOR_SCHEMA(ReduceBoxesOnly)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Reduce the boxes from the same image.
)DOC")
    .Arg(
        "im",
        "(int) image index.")
    .Arg(
        "dpi",
        "(int) number of detections per image.")
    .Input(
        0,
        "boxes_in",
        "bounding box information of shape (R, 12): x1, y1, x2, y2, score, area, c, im, lvl, a, h, w.")
    .Output(
        0,
        "rois",
        "2D tensor of shape (R, 11): im, x1, y1, x2, y2, lvl, c, lvl, a, h, w.")
    .Output(
        1,
        "stats",
        "statistics for each level, of shape (num_levels)");

SHOULD_NOT_DO_GRADIENT(ReduceBoxesOnly);

} // namespace caffe2