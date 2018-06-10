#include "compute_box_targets_op.h"

namespace caffe2 {

namespace {

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

void _ones(const int size, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = 1.;
    }
}

void _compute_targets(const float x1A, const float y1A, const float x2A, const float y2A,
                    const float x1B, const float y1B, const float x2B, const float y2B,
                    const int offset, 
                    float* bbox_targets_pointer,
                    float* bbox_inside_weights_pointer,
                    float* bbox_outside_weights_pointer) {
  const float ex_width = x2A - x1A + 1.;
  const float ex_height = y2A - y1A + 1.;
  const float ex_ctr_x = x1A + 0.5 * ex_width;
  const float ex_ctr_y = y1A + 0.5 * ex_height;

  const float gt_width = x2B - x1B + 1.;
  const float gt_height = y2B - y1B + 1.;
  const float gt_ctr_x = x1B + 0.5 * gt_width;
  const float gt_ctr_y = y1B + 0.5 * gt_height;

  // const float wx = 10.;
  // const float wy = 10.;
  // const float ww = 5.;
  // const float wh = 5.;

  const float target_dx = 10. * (gt_ctr_x - ex_ctr_x) / ex_width;
  const float target_dy = 10. * (gt_ctr_y - ex_ctr_y) / ex_height;
  const float target_dw = 5. * log(gt_width / ex_width);
  const float target_dh = 5. * log(gt_height / ex_height);

  bbox_targets_pointer[offset] = target_dx;
  bbox_targets_pointer[offset+1] = target_dy;
  bbox_targets_pointer[offset+2] = target_dw;
  bbox_targets_pointer[offset+3] = target_dh;

  _ones(4, bbox_inside_weights_pointer + offset);
  _ones(4, bbox_outside_weights_pointer + offset);
}

}

template<>
bool ComputeBoxTargetsOp<float, CPUContext>::RunOnDevice() {
  auto& rois = Input(0);
  auto& gt_boxes = Input(1);
  auto& labels = Input(2);
  auto& targets = Input(3);

  const float* rois_pointer = rois.data<float>();
  const float* gt_boxes_pointer = gt_boxes.data<float>();
  const int* labels_pointer = labels.data<int>();
  const int* targets_pointer = targets.data<int>();

  const int this_cnt = rois.dim32(0);

  auto* bbox_targets = Output(0);
  auto* bbox_inside_weights = Output(1);
  auto* bbox_outside_weights = Output(2);

  bbox_targets->Resize(this_cnt, 4 * num_classes_);
  bbox_inside_weights->Resize(this_cnt, 4 * num_classes_);
  bbox_outside_weights->Resize(this_cnt, 4 * num_classes_);

  float* bbox_targets_pointer = bbox_targets->mutable_data<float>();
  float* bbox_inside_weights_pointer = bbox_inside_weights->mutable_data<float>();
  float* bbox_outside_weights_pointer = bbox_outside_weights->mutable_data<float>();
  
  // set to zeros
  const int total_cnt = this_cnt * (4 * num_classes_);
  _zeros(total_cnt, bbox_targets_pointer);
  _zeros(total_cnt, bbox_inside_weights_pointer);
  _zeros(total_cnt, bbox_outside_weights_pointer);

  for (int i=0; i<this_cnt; i++) {
    const int this_label = labels_pointer[i];
    if (this_label > 0) {
      const int gt_id = targets_pointer[i];
      const int Ap = i * 5;
      const int Bp = gt_id * 5;
      _compute_targets(rois_pointer[Ap+1], rois_pointer[Ap+2], 
                      rois_pointer[Ap+3], rois_pointer[Ap+4],
                    gt_boxes_pointer[Bp], gt_boxes_pointer[Bp+1], 
                    gt_boxes_pointer[Bp+2], gt_boxes_pointer[Bp+3],
                    (i * num_classes_ + this_label) * 4, 
                    bbox_targets_pointer,
                    bbox_inside_weights_pointer,
                    bbox_outside_weights_pointer);
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(ComputeBoxTargets, ComputeBoxTargetsOp<float, CPUContext>);

OPERATOR_SCHEMA(ComputeBoxTargets)
    .NumInputs(4)
    .NumOutputs(3)
    .SetDoc(R"DOC(
An operate to just compute the bounding box targets.
)DOC")
    .Input(
        0,
        "rois",
        "sampled rois for training.")
    .Input(
        1,
        "gt_boxes",
        "ground truth boxes, with shape (G, 4).")
    .Input(
        2,
        "labels",
        "labels for the rois, type int32.")
    .Input(
        3,
        "targets",
        "target ground truth being assigned.")
    .Output(
        0,
        "bbox_targets",
        "bounding box regression targets.")
    .Output(
        1,
        "bbox_inside_weights",
        "bounding box inside weights.")
    .Output(
        2,
        "bbox_outside_weights",
        "bounding box outside weights.")
    ;


SHOULD_NOT_DO_GRADIENT(ComputeBoxTargets);

} // namespace caffe2