#include "class_based_boxes_combined_op.h"

namespace caffe2 {

namespace {

void _copy(const int size, const float* source, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

}

template<>
bool ClassBasedBoxesCombinedOp<float, CPUContext>::RunOnDevice() {
  auto& stats = Input(0);
  const int num_cls = stats.dim32(0);
  auto& boxes = Input(1);
  const int num_total = boxes.dim32(0);
  DCHECK_EQ(boxes.dim32(1), 12);
  const float* boxes_pointer = boxes.data<float>();
  // get the statistics to cpu for the current class
  const int* stats_pointer = stats.data<int>();
  DCHECK_EQ(OutputSize(), num_cls);
  float* cls_boxes_pointers[num_cls];

  for (int cls=0; cls<num_cls; cls++) {
    int current_cls_count = stats_pointer[cls];
    // use the stats to initialize class based tensors
    auto* cls_boxes = Output(cls);
    cls_boxes->Resize(current_cls_count, 13);
    cls_boxes_pointers[cls] = cls_boxes->mutable_data<float>();
  }
  
  for (int i=0, bp=0; i<num_total; i++, bp+=12) {
    const float valid = boxes_pointer[bp+6];
    if (valid > 0.) {
      // returns the old index
      const float cls_f = boxes_pointer[bp+5];
      const int cls = static_cast<int>(cls_f);
      float* cls_pointer = cls_boxes_pointers[cls];
      const float x1 = boxes_pointer[bp];
      const float y1 = boxes_pointer[bp+1];
      const float x2 = boxes_pointer[bp+2];
      const float y2 = boxes_pointer[bp+3];

      *(cls_pointer++) = x1;
      *(cls_pointer++) = y1;
      *(cls_pointer++) = x2;
      *(cls_pointer++) = y2;
      *(cls_pointer++) = boxes_pointer[bp+4];
      *(cls_pointer++) = (x2 - x1 + 1.) * (y2 - y1 + 1.);
      *(cls_pointer++) = cls_f;
      // leave a dimension to encode suppressed information
      *(cls_pointer++) = 0.;
      // just copy from now on
      _copy(5, boxes_pointer + (bp+7), cls_pointer);

      cls_boxes_pointers[cls] += 13;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(ClassBasedBoxesCombined, ClassBasedBoxesCombinedOp<float, CPUContext>);

OPERATOR_SCHEMA(ClassBasedBoxesCombined)
    .NumInputs(2)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Return class based boxes, no features.
)DOC")
    .Input(
        0,
        "stats",
        "(int) aggregated statistics of shape (num_cls).")
    .Input(
        1,
        "boxes",
        "bounding box information of shape (R, 12): x1, y1, x2, y2, score, c, valid, im, lvl, a, h, w.")
    .Output(
        0,
        "boxes_cls1",
        "bounding box information for class 1 of shape (T, 13): x1, y1, x2, y2, score, area, c, ZERO, im, lvl, a, h, w.");;


SHOULD_NOT_DO_GRADIENT(ClassBasedBoxesCombined);

} // namespace caffe2