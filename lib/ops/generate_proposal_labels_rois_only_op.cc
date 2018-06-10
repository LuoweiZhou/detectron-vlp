#include "generate_proposal_labels_rois_only_op.h"

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

void _minus_ones(const int size, int* target) {
    for (int i=0; i<size; i++) {
        *(target++) = -1;
    }
}

void _zeros(const int size, int* target) {
    for (int i=0; i<size; i++) {
        *(target++) = 0;
    }
}

}

template<>
bool GenerateProposalLabelsRoIsOnlyOp<float, CPUContext>::RunOnDevice() {
  auto& rpn_rois = Input(0);
  auto& gt_boxes = Input(1);
  auto& gt_classes = Input(2);

  const float* rpn_rois_pointer = rpn_rois.data<float>();
  const float* gt_boxes_pointer = gt_boxes.data<float>();
  const int* gt_classes_pointer = gt_classes.data<int>();

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
  auto* targets = Output(2);

  rois->Resize(this_cnt, 5);
  labels->Resize(this_cnt);
  targets->Resize(this_cnt);

  float* rois_pointer = rois->mutable_data<float>();
  int* labels_pointer = labels->mutable_data<int>();
  int* targets_pointer = targets->mutable_data<int>();
  // set to zeros
  _zeros(this_cnt, labels_pointer);
  _minus_ones(this_cnt, targets_pointer);

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
    targets_pointer[i] = gt_id;
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

REGISTER_CPU_OPERATOR(GenerateProposalLabelsRoIsOnly, GenerateProposalLabelsRoIsOnlyOp<float, CPUContext>);

OPERATOR_SCHEMA(GenerateProposalLabelsRoIsOnly)
    .NumInputs(3)
    .NumOutputs(3)
    .SetDoc(R"DOC(
Sample the region proposals and return the rois and the labels only, do not care bounding boxes.
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
        "targets",
        "ground truth target id.")
    ;


SHOULD_NOT_DO_GRADIENT(GenerateProposalLabelsRoIsOnly);

} // namespace caffe2