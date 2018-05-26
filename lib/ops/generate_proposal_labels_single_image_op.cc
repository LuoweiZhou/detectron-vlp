#include "generate_proposal_labels_single_image_op.h"

namespace caffe2 {

namespace {

}

template<>
bool GenerateProposalLabelsSingleImageOp<float, CPUContext>::RunOnDevice() {
  auto& rpn_rois = Input(0);
  auto& gt_boxes = Input(1);
  auto& gt_classes = Input(2);
  auto& im_info = Input(3);

  auto* rois = Output(0);
  auto* labels = Output(1);
  auto* bbox_targets = Output(2);
  auto* bbox_inside_weights = Output(3);
  auto* bbox_outside_weights = Output(4);

  rois->Resize(rois_per_image_, 5);
  labels->Resize(rois_per_image_);
  bbox_targets->Resize(rois_per_image_, 4);
  bbox_inside_weights->Resize(rois_per_image_, 4);
  bbox_outside_weights->Resize(rois_per_image_, 4);

  // const int num_inputs = InputSize();
  // auto& cls_probs = Input(0);
  // auto& box_preds = Input(1);
  // auto& anchors = Input(2);
  // auto& im_info = Input(3);

  // // get some sizes and pointers
  // const int N = cls_probs.dim32(0);
  // DCHECK_EQ(N, box_preds.dim32(0));
  // DCHECK_EQ(N, im_info.dim32(0));
  // const int A = cls_probs.dim32(1);
  // DCHECK_EQ(A * 4, box_preds.dim32(1));
  // DCHECK_EQ(A, anchors.dim32(0));
  // const int H = cls_probs.dim32(2);
  // const int W = cls_probs.dim32(3);
  // const int P = H * W;
  // const int num_probs = A * P;
  // const float* cls_prob_pointer = cls_probs.data<float>() + im_ * num_probs;
  // const float* box_pred_pointer = box_preds.data<float>() + im_ * (4 * num_probs);
  // const float* anchor_pointer = anchors.data<float>();
  // const float* im_info_pointer = im_info.data<float>() + im_ * 3;
  // const float height_max = im_info_pointer[0] - 1.;
  // const float width_max = im_info_pointer[1] - 1.;

  // // get the top locations
  // int R;
  // int* yi_data;
  // float* yv_data;
  // if (num_probs <= pre_top_n_) {
  //   R = num_probs;
  //   // just select everything
  //   Yi.Resize(R);
  //   Yv.Resize(R);
  //   // copy index
  //   yi_data = Yi.mutable_data<int>();
  //   for (int i=0; i<R; i++) {
  //     yi_data[i] = static_cast<int>(i);
  //   }
  //   yv_data = Yv.mutable_data<float>();
  //   // copy values
  //   context_.Copy<float, CPUContext, CPUContext>(R, cls_prob_pointer, yv_data);
  // } else {
  //   R = pre_top_n_;
  //   // only get the top ones
  //   Yi.Resize(R);
  //   Yv.Resize(R);
  //   yi_data = Yi.mutable_data<int>();
  //   yv_data = Yv.mutable_data<float>();
  //   // build a priority queue
  //   priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>> PQ;
  //   _build_heap(&PQ, cls_prob_pointer, num_probs, R);
  //   for (int i=0; i<R; i++) {
  //     auto& pqelm = PQ.top();
  //     yv_data[i] = pqelm.first;
  //     yi_data[i] = static_cast<int>(pqelm.second);
  //     PQ.pop();
  //   }
  // }
  
  // // then get the boxes
  // // x1, y1, x2, y2, area, invalid/suppressed
  // rois_raw.Resize(R, 6);
  // vector<pair<float, int>> index_list;
  // float* rois_raw_pointer = rois_raw.mutable_data<float>();
  // for (int r=0, bbp=0; r<R; r++, bbp+=6) {
  //   const int index = yi_data[r];
  //   int ind = index;
  //   const int w = ind % W;
  //   ind /= W;
  //   const int h = ind % H;
  //   const int a = ind / H;
  //   const int ap = a * 4;
  //   const int bp = (a * 4 * H + h) * W + w;

  //   const float x = w * stride_;
  //   const float y = h * stride_;

  //   const float x1 = x + anchor_pointer[ap];
  //   const float y1 = y + anchor_pointer[ap+1];
  //   const float x2 = x + anchor_pointer[ap+2];
  //   const float y2 = y + anchor_pointer[ap+3];

  //   const float dx = box_pred_pointer[bp];
  //   const float dy = box_pred_pointer[bp + P];
  //   float dw = box_pred_pointer[bp + 2 * P];
  //   float dh = box_pred_pointer[bp + 3 * P];
  //   dw = _clip_max(dw, 4.1351666);
  //   dh = _clip_max(dh, 4.1351666);

  //   // do box transform
  //   const float ww = x2 - x1 + 1.;
  //   const float hh = y2 - y1 + 1.;
  //   const float ctr_x = x1 + 0.5 * ww;
  //   const float ctr_y = y1 + 0.5 * hh;

  //   const float pred_ctr_x = dx * ww + ctr_x;
  //   const float pred_ctr_y = dy * hh + ctr_y;
  //   float pred_w = exp(dw) * ww;
  //   float pred_h = exp(dh) * hh;

  //   const float xx1 = _clip_max(_clip_min(pred_ctr_x - 0.5 * pred_w, 0.), width_max);
  //   const float yy1 = _clip_max(_clip_min(pred_ctr_y - 0.5 * pred_h, 0.), height_max);
  //   const float xx2 = _clip_max(_clip_min(pred_ctr_x + 0.5 * pred_w - 1., 0.), width_max);
  //   const float yy2 = _clip_max(_clip_min(pred_ctr_y + 0.5 * pred_h - 1., 0.), height_max); 

  //   pred_w = xx2 - xx1 + 1.;
  //   pred_h = yy2 - yy1 + 1.;

  //   rois_raw_pointer[bbp] = xx1;
  //   rois_raw_pointer[bbp+1] = yy1;
  //   rois_raw_pointer[bbp+2] = xx2;
  //   rois_raw_pointer[bbp+3] = yy2;
  //   rois_raw_pointer[bbp+4] = pred_w * pred_h;
  //   // not suppressed
  //   rois_raw_pointer[bbp+5] = (pred_w < 1. || pred_h < 1.) ? 1. : 0.;
  //   index_list.push_back(make_pair(yv_data[r], r));
  // }

  // sort(index_list.begin(), index_list.end(), _compare_value<float>());
  // // then get nms
  // int cnt = 0;
  // for (int i=0; i<R; i++) {
  //   const int ind = index_list[i].second;
  //   const int bp = ind * 6;
  //   // if not suppressed
  //   if (rois_raw_pointer[bp+5] == 0.) {
  //       // leave it untouched
  //       const float x1A = rois_raw_pointer[bp];
  //       const float y1A = rois_raw_pointer[bp+1];
  //       const float x2A = rois_raw_pointer[bp+2];
  //       const float y2A = rois_raw_pointer[bp+3];
  //       const float areaA = rois_raw_pointer[bp+4]; 
  //       // suppress others
  //       for (int j=i+1; j<R; j++) {
  //           const int jnd = index_list[j].second;
  //           const int bbp = jnd * 6;
  //           const float x1B = rois_raw_pointer[bbp];
  //           const float y1B = rois_raw_pointer[bbp+1];
  //           const float x2B = rois_raw_pointer[bbp+2];
  //           const float y2B = rois_raw_pointer[bbp+3];
  //           const float areaB = rois_raw_pointer[bbp+4];

  //           const float xx1 = (x1A > x1B) ? x1A : x1B;
  //           const float yy1 = (y1A > y1B) ? y1A : y1B;
  //           const float xx2 = (x2A < x2B) ? x2A : x2B;
  //           const float yy2 = (y2A < y2B) ? y2A : y2B;

  //           float w = xx2 - xx1 + 1.;
  //           w = (w > 0.) ? w : 0.;
  //           float h = yy2 - yy1 + 1.;
  //           h = (h > 0.) ? h : 0.;
  //           const float inter = w * h;
  //           const float iou = inter / (areaA + areaB - inter);

  //           if (iou >= nms_) {
  //               rois_raw_pointer[bbp+5] = 1.;
  //           }
  //       }
  //       cnt ++; 
  //   }
  //   // enough boxes
  //   if (cnt == post_top_n_) {
  //       for (int j=i+1; j<R; j++) {
  //           const int jnd = index_list[j].second;
  //           rois_raw_pointer[jnd*6+5] = 1.;
  //       }
  //       break;
  //   }
  // }

  // // then copy it out
  // auto* rois = Output(0);
  // auto* roi_probs = Output(1);
  // rois->Resize(cnt, 5);
  // roi_probs->Resize(cnt, 1);
  // float* rois_pointer = rois->mutable_data<float>();
  // float* roi_probs_pointer = roi_probs->mutable_data<float>();

  // int n = 0;
  // for (int i=0; i<R; i++) {
  //   const int ind = index_list[i].second;
  //   const int bp = ind * 6;
  //   if (!rois_raw_pointer[bp+5]) {
  //     // copy some data
  //     const int bbp = n * 5;
  //     rois_pointer[bbp] = static_cast<float>(im_);
  //     rois_pointer[bbp+1] = rois_raw_pointer[bp];
  //     rois_pointer[bbp+2] = rois_raw_pointer[bp+1];
  //     rois_pointer[bbp+3] = rois_raw_pointer[bp+2];
  //     rois_pointer[bbp+4] = rois_raw_pointer[bp+3];
  //     roi_probs_pointer[n] = index_list[i].first;
  //     n++;
  //   }
  // }

  // DCHECK_EQ(n, cnt);
  // index_list.clear();

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