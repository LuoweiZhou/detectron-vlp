#include "generate_proposal_labels_single_image_op.h"

using std::vector;
using std::pair;
using std::make_pair;
using std::shuffle;

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

void _zeros(const int size, int* target) {
    for (int i=0; i<size; i++) {
        *(target++) = 0;
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
bool GenerateProposalLabelsSingleImageOp<float, CPUContext>::RunOnDevice() {
  auto& rpn_rois = Input(0);
  auto& gt_boxes = Input(1);
  auto& gt_classes = Input(2);
  auto& im_info = Input(3);

  const float* rpn_rois_pointer = rpn_rois.data<float>();
  // there should be at least one proposal I think
  DCHECK_EQ(im_, static_cast<int>(rpn_rois_pointer[0]));
  const float* gt_boxes_pointer = gt_boxes.data<float>();
  const int* gt_classes_pointer = gt_classes.data<int>();
  const float* im_info_pointer = im_info.data<float>() + im_ * 3;
  const float im_scale = im_info_pointer[2];

  const int R = rpn_rois.dim32(0);
  const int G = gt_boxes.dim32(0);

  // build candidates
  vector<pair<int, int>> fg_candidates;
  vector<pair<int, int>> bg_candidates;
  for (int i=0; i<G; i++) {
    fg_candidates.push_back(make_pair(i, i));
  }
  for (int i=0; i<R; i++) {
    const int Ap = i * 5;
    const float x1A = rpn_rois_pointer[Ap+1];
    const float y1A = rpn_rois_pointer[Ap+2];
    const float x2A = rpn_rois_pointer[Ap+3];
    const float y2A = rpn_rois_pointer[Ap+4];
    const float areaA = (x2A - x1A + 1.) * (y2A - y1A + 1.);
    float max_iou = 0.;
    int gt_assign = -1;
    for (int j=0; j<G; j++) {
      const int Bp = j * 5;
      const float x1B = gt_boxes_pointer[Bp];
      const float y1B = gt_boxes_pointer[Bp+1];
      const float x2B = gt_boxes_pointer[Bp+2];
      const float y2B = gt_boxes_pointer[Bp+3];
      const float areaB = gt_boxes_pointer[Bp+4];

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

      if (iou > max_iou) {
        max_iou = iou;
        gt_assign = j;
      }
    }
    // assign it to different modules
    if (max_iou >= fg_thresh_) {
      fg_candidates.push_back(make_pair(i+G, gt_assign));
    }
    if (max_iou < bg_thresh_hi_ && max_iou >= bg_thresh_lo_) {
      bg_candidates.push_back(make_pair(i, -1));
    }
  }
  const int fg_cnt = fg_candidates.size();
  const int bg_cnt = bg_candidates.size();

  // then sample the candidates, conditionally shuffle the candidates
  int this_fg_cnt;
  if (fg_cnt < fg_rois_per_image_) {
    this_fg_cnt = fg_cnt;
  } else {
    this_fg_cnt = fg_rois_per_image_;
    shuffle(fg_candidates.begin(), fg_candidates.end(), rng_);
  }
  int this_bg_cnt;
  if (bg_cnt < (rois_per_image_ - this_fg_cnt)) {
    this_bg_cnt = bg_cnt;
  } else {
    this_bg_cnt = (rois_per_image_ - this_fg_cnt);
    shuffle(bg_candidates.begin(), bg_candidates.end(), rng_);
  }
  const int this_cnt = this_fg_cnt + this_bg_cnt;

  auto* rois = Output(0);
  auto* labels = Output(1);
  auto* bbox_targets = Output(2);
  auto* bbox_inside_weights = Output(3);
  auto* bbox_outside_weights = Output(4);

  rois->Resize(this_cnt, 5);
  labels->Resize(this_cnt);
  bbox_targets->Resize(this_cnt, 4 * num_classes_);
  bbox_inside_weights->Resize(this_cnt, 4 * num_classes_);
  bbox_outside_weights->Resize(this_cnt, 4 * num_classes_);

  float* rois_pointer = rois->mutable_data<float>();
  int* labels_pointer = labels->mutable_data<int>();
  float* bbox_targets_pointer = bbox_targets->mutable_data<float>();
  float* bbox_inside_weights_pointer = bbox_inside_weights->mutable_data<float>();
  float* bbox_outside_weights_pointer = bbox_outside_weights->mutable_data<float>();
  // set to zeros
  _zeros(this_cnt, labels_pointer);
  const int total_cnt = this_cnt * (4 * num_classes_);
  _zeros(total_cnt, bbox_targets_pointer);
  _zeros(total_cnt, bbox_inside_weights_pointer);
  _zeros(total_cnt, bbox_outside_weights_pointer);

  // then copy it to the outputs
  for (int i=0; i<this_fg_cnt; i++) {
    const int this_id = fg_candidates[i].first;
    const int gt_id = fg_candidates[i].second;
    float x1A, y1A, x2A, y2A;

    if (this_id < G) {
      const int Ap = this_id * 5;
      x1A = gt_boxes_pointer[Ap];
      y1A = gt_boxes_pointer[Ap+1];
      x2A = gt_boxes_pointer[Ap+2];
      y2A = gt_boxes_pointer[Ap+3];
    } else {
      const int Ap = (this_id - G) * 5;
      x1A = rpn_rois_pointer[Ap+1];
      y1A = rpn_rois_pointer[Ap+2];
      x2A = rpn_rois_pointer[Ap+3];
      y2A = rpn_rois_pointer[Ap+4];
    }

    const int Rp = i * 5;
    rois_pointer[Rp] = static_cast<float>(im_);
    rois_pointer[Rp+1] = x1A;
    rois_pointer[Rp+2] = y1A;
    rois_pointer[Rp+3] = x2A;
    rois_pointer[Rp+4] = y2A;

    const int this_label = gt_classes_pointer[gt_id];
    labels_pointer[i] = this_label;

    const int Bp = gt_id * 5;
    const float x1B = gt_boxes_pointer[Bp];
    const float y1B = gt_boxes_pointer[Bp+1];
    const float x2B = gt_boxes_pointer[Bp+2];
    const float y2B = gt_boxes_pointer[Bp+3];

    _compute_targets(x1A, y1A, x2A, y2A,
                    x1B, y1B, x2B, y2B,
                    (i * num_classes_ + this_label) * 4, 
                    bbox_targets_pointer,
                    bbox_inside_weights_pointer,
                    bbox_outside_weights_pointer);
  }

  for (int i=0; i<this_bg_cnt; i++) {
    const int this_id = bg_candidates[i].first;
    const int Ap = this_id * 5;
    const int Rp = (i + this_fg_cnt) * 5;
    _copy(5, rpn_rois_pointer + Ap, rois_pointer + Rp);
  }

  fg_candidates.clear();
  bg_candidates.clear();

  return true;
}

REGISTER_CPU_OPERATOR(GenerateProposalLabelsSingleImage, GenerateProposalLabelsSingleImageOp<float, CPUContext>);

OPERATOR_SCHEMA(GenerateProposalLabelsSingleImage)
    .NumInputs(4)
    .NumOutputs(5)
    .SetDoc(R"DOC(
Sample the region proposals and return their labels.
)DOC")
    .Arg(
        "rois_per_image",
        "(int) number of rois per image.")
    .Arg(
        "fg_rois_per_image",
        "(int) targeted foreground rois per image.")
    .Arg(
        "fg_thresh",
        "(float) iou threshold above which to be count as foreground.")
    .Arg(
        "bg_thresh_hi",
        "(float) iou threshold below which to be count as background.")
    .Arg(
        "bg_thresh_lo",
        "(float) iou threshold above which to be count as background.")
    .Arg(
        "im",
        "(int) index of the image.")
    .Input(
        0,
        "rpn_rois",
        "rois from region proposals, scaled, with shape (R, 5).")
    .Input(
        1,
        "gt_boxes",
        "ground truth boxes, with shape (G, 4).")
    .Input(
        2,
        "gt_classes",
        "ground truth classes, with shape (G,).")
    .Input(
        3,
        "im_info",
        "image information, with (N, 3) shape.")
    .Output(
        0,
        "rois",
        "sampled rois for training.")
    .Output(
        1,
        "labels",
        "labels for the rois, type int32.")
    .Output(
        2,
        "bbox_targets",
        "bounding box regression targets.")
    .Output(
        3,
        "bbox_inside_weights",
        "bounding box inside weights.")
    .Output(
        4,
        "bbox_outside_weights",
        "bounding box outside weights.")
    ;


SHOULD_NOT_DO_GRADIENT(GenerateProposalLabelsSingleImage);

} // namespace caffe2