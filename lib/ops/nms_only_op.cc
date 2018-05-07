#include "nms_only_op.h"

using std::pair;
using std::make_pair;
using std::sort;
using std::vector;

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

void _copy(const int size, const float* source, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

}

template<>
bool NMSOnlyOp<float, CPUContext>::RunOnDevice() {
  auto& boxes = Input(0);
  DCHECK_EQ(boxes.dim32(1), 13);
  const float* boxes_pointer = boxes.data<float>();
  const int num_total = boxes.dim32(0);

  // handle the empty case and the case when there is a single box
  if (num_total == 0) {
    Output(0)->Resize(0, 12);
    Output(0)->mutable_data<float>();
    return true;
  } else if (num_total == 1) {
    auto* output_boxes = Output(0);
    output_boxes->Resize(1, 12);
    float* output_boxes_pointer = output_boxes->mutable_data<float>();
    // first copy the boxes
    _copy(7, boxes_pointer, output_boxes_pointer);
    _copy(5, boxes_pointer + 8, output_boxes_pointer + 7);
    return true;
  }

  // sort the class scores 
  vector<pair<float, int>> index_list;
  int suppressed[num_total];
  for (int i=0; i<num_total; i++) {
    suppressed[i] = 0;
    index_list.push_back(make_pair(boxes_pointer[i*13+4], i));
  }
  sort(index_list.begin(), index_list.end(), _compare_value<float>());

  // then go down the list and do nms
  int cnt = 0;
  for (int i=0; i<num_total; i++) {
    const int ind = index_list[i].second;
    // if not suppressed
    if (suppressed[ind] == 0) {
        // leave it untouched
        const int bp = ind * 13;
        const float x1A = boxes_pointer[bp];
        const float y1A = boxes_pointer[bp+1];
        const float x2A = boxes_pointer[bp+2];
        const float y2A = boxes_pointer[bp+3];
        const float areaA = boxes_pointer[bp + 5]; 
        // suppress others
        for (int j=i+1; j<num_total; j++) {
            const int jnd = index_list[j].second;
            const int bbp = jnd * 13;
            const float x1B = boxes_pointer[bbp];
            const float y1B = boxes_pointer[bbp+1];
            const float x2B = boxes_pointer[bbp+2];
            const float y2B = boxes_pointer[bbp+3];
            const float areaB = boxes_pointer[bbp + 5];

            const float xx1 = (x1A > x1B) ? x1A : x1B;
            const float yy1 = (y1A > y1B) ? y1A : y1B;
            const float xx2 = (x2A < x2B) ? x2A : x2B;
            const float yy2 = (y2A < y2B) ? y2A : y2B;

            float w = xx2 - xx1 + 1.;
            w = (w > 0.) ? w : 0.;
            float h = yy2 - yy1 + 1.;
            h = (h > 0.) ? h : 0.;
            const float inter = w * h;
            const float iou = inter / (areaA + areaB - inter);

            if (iou >= nms_) {
                suppressed[jnd] = 1;
            }
        }
        cnt ++; 
    }
    // enough boxes
    if (cnt == dpi_) {
        for (int j=i+1; j<num_total; j++) {
            const int jnd = index_list[j].second;
            suppressed[jnd] = 1;
        }
        break;
    }
  }
  index_list.clear();

  auto* out_boxes = Output(0);
  out_boxes->Resize(cnt, 12);
  float* out_boxes_pointer = out_boxes->mutable_data<float>();

  // copy the boxes
  int n = 0;
  for (int i=0; i<num_total; i++) {
    if (!suppressed[i]) {
        const int bp = i * 13;
        const int bbp = n * 12;
        _copy(7, boxes_pointer + bp, out_boxes_pointer + bbp);
        _copy(5, boxes_pointer + bp + 8, out_boxes_pointer + bbp + 7);
        n++;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(NMSOnly, NMSOnlyOp<float, CPUContext>);

OPERATOR_SCHEMA(NMSOnly)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Do non maximal suppression and return at most dpi boxes.
)DOC")
    .Arg(
        "nms",
        "(float) NMS threshold.")
    .Arg(
        "dpi",
        "(int) number of detections per image.")
    .Input(
        0,
        "boxes",
        "bounding box information of shape (R, 13): x1, y1, x2, y2, score, area, c, ZERO, im, lvl, a, h, w.")
    .Output(
        0,
        "boxes_out",
        "bounding box information of shape (Rx, 12): x1, y1, x2, y2, score, area, c, im, lvl, a, h, w.");

SHOULD_NOT_DO_GRADIENT(NMSOnly);

} // namespace caffe2