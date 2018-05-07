#include "class_based_boxes_only_op.h"

namespace caffe2 {

namespace {

void _copy(const int size, const float* source, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

}

template<>
bool ClassBasedBoxesOnlyOp<float, CPUContext>::RunOnDevice() {
  auto& stats = Input(0);
  const int num_cls = stats.dim32(0);
  DCHECK_LT(class_, num_cls);
  auto& boxes = Input(1);
  const int num_total = boxes.dim32(0);
  DCHECK_EQ(boxes.dim32(1), 12);
  const float* boxes_pointer = boxes.data<float>();

  // get the statistics to cpu for the current class
  const int* stats_pointer = stats.data<int>();
  int current_cls_count = stats_pointer[class_];
  // use the stats to initialize class based tensors
  auto* cls_boxes = Output(0);
  cls_boxes->Resize(current_cls_count, 13);
  float* cls_boxes_pointer = cls_boxes->mutable_data<float>();

  if (current_cls_count > 0) {
    int bbp = 0;
    for (int i=0, bp=0; i<num_total; i++, bp+=12) {
        const float cls_f = boxes_pointer[bp+5];
        const float valid = boxes_pointer[bp+6];
        if (cls_f == class_float_ && valid > 0.) {
          // returns the old index

          const float x1 = boxes_pointer[bp];
          const float y1 = boxes_pointer[bp+1];
          const float x2 = boxes_pointer[bp+2];
          const float y2 = boxes_pointer[bp+3];

          cls_boxes_pointer[bbp] = x1;
          cls_boxes_pointer[bbp+1] = y1;
          cls_boxes_pointer[bbp+2] = x2;
          cls_boxes_pointer[bbp+3] = y2;
          cls_boxes_pointer[bbp+4] = boxes_pointer[bp+4];
          cls_boxes_pointer[bbp+5] = (x2 - x1 + 1.) * (y2 - y1 + 1.);
          cls_boxes_pointer[bbp+6] = cls_f;
          // leave a dimension to encode suppressed information
          cls_boxes_pointer[bbp+7] = 0.;
          // just copy from now on
          _copy(5, boxes_pointer + (bp+7), cls_boxes_pointer + (bbp+8));

          bbp += 13;
        }
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(ClassBasedBoxesOnly, ClassBasedBoxesOnlyOp<float, CPUContext>);

OPERATOR_SCHEMA(ClassBasedBoxesOnly)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Return class based boxes, no features.
)DOC")
    .Arg(
        "class_id",
        "(int) the class to consider in this op, 0-based.")
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
        "boxes_cls",
        "bounding box information for class 1 of shape (T, 13): x1, y1, x2, y2, score, area, c, ZERO, im, lvl, a, h, w.");;


SHOULD_NOT_DO_GRADIENT(ClassBasedBoxesOnly);

} // namespace caffe2