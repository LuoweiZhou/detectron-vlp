#include "generate_proposals_single_image_op.h"

using std::pair;
using std::make_pair;
using std::sort;
using std::vector;
using std::priority_queue;

namespace caffe2 {

namespace {

float _clip_min(const float f, const float m) {
    return (f > m) ? (f) : (m);
}

float _clip_max(const float f, const float m) {
    return (f < m) ? (f) : (m);
}

void _zeros(const int size, int* p) {
    for (int i=0; i<size; i++) {
        *(p++) = 0;
    }
}

template <typename T>
struct _compare_value {
  bool operator()(
      const pair<T, int>& lhs,
      const pair<T, int>& rhs) {
    return (lhs.first > rhs.first);
  }
};

int _build_heap(priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>>* PQ,
                const float* cls_prob, const int num_probs, const int top_n) {
    for (int i=0; i<num_probs; i++) {
        const float prob = cls_prob[i];
        if (PQ->size() < top_n || prob > PQ->top().first) {
            PQ->push(make_pair(prob, i));
        }
        if (PQ->size() > top_n) {
            PQ->pop();
        }
    }
    return PQ->size();
}

}

template<>
bool GenerateProposalsSingleImageOp<float, CPUContext>::RunOnDevice() {
  // const int num_inputs = InputSize();
  auto& cls_probs = Input(0);
  auto& box_preds = Input(1);
  auto& anchors = Input(2);
  auto& im_info = Input(3);

  // get some sizes and pointers
  const int N = cls_probs.dim32(0);
  DCHECK_EQ(N, box_preds.dim32(0));
  DCHECK_EQ(N, im_info.dim32(0));
  const int A = cls_probs.dim32(1);
  DCHECK_EQ(A * 4, box_preds.dim32(1));
  DCHECK_EQ(A, anchors.dim32(0));
  const int H = cls_probs.dim32(2);
  const int W = cls_probs.dim32(3);
  const int P = H * W;
  const int num_probs = A * P;
  const float* cls_prob_pointer = cls_probs.data<float>() + im_ * num_probs;
  const float* box_pred_pointer = box_preds.data<float>() + im_ * (4 * num_probs);
  const float* anchor_pointer = anchors.data<float>();
  const float* im_info_pointer = im_info.data<float>() + im_ * 3;
  const float height_max = im_info_pointer[0] - 1.;
  const float width_max = im_info_pointer[1] - 1.;

  // get the top locations
  int R;
  int* yi_data;
  float* yv_data;
  if (num_probs <= pre_top_n_) {
    R = num_probs;
    // just select everything
    Yi.Resize(R);
    Yv.Resize(R);
    // copy index
    yi_data = Yi.mutable_data<int>();
    for (int i=0; i<R; i++) {
      yi_data[i] = static_cast<int>(i);
    }
    yv_data = Yv.mutable_data<float>();
    // copy values
    context_.Copy<float, CPUContext, CPUContext>(R, cls_prob_pointer, yv_data);
  } else {
    R = pre_top_n_;
    // only get the top ones
    Yi.Resize(R);
    Yv.Resize(R);
    yi_data = Yi.mutable_data<int>();
    yv_data = Yv.mutable_data<float>();
    // build a priority queue
    priority_queue<pair<float, int>, vector<pair<float, int>>, _compare_value<float>> PQ;
    _build_heap(&PQ, cls_prob_pointer, num_probs, R);
    for (int i=0; i<R; i++) {
      auto& pqelm = PQ.top();
      yv_data[i] = pqelm.first;
      yi_data[i] = static_cast<int>(pqelm.second);
      PQ.pop();
    }
  }
  
  // then get the boxes
  // x1, y1, x2, y2, area, invalid/suppressed
  rois_raw.Resize(R, 6);
  vector<pair<float, int>> index_list;
  float* rois_raw_pointer = rois_raw.mutable_data<float>();
  for (int r=0, bbp=0; r<R; r++, bbp+=6) {
    const int index = yi_data[r];
    int ind = index;
    const int w = ind % W;
    ind /= W;
    const int h = ind % H;
    const int a = ind / H;
    const int ap = a * 4;
    const int bp = (a * 4 * H + h) * W + w;

    const float x = w * stride_;
    const float y = h * stride_;

    const float x1 = x + anchor_pointer[ap];
    const float y1 = y + anchor_pointer[ap+1];
    const float x2 = x + anchor_pointer[ap+2];
    const float y2 = y + anchor_pointer[ap+3];

    const float dx = box_pred_pointer[bp];
    const float dy = box_pred_pointer[bp + P];
    float dw = box_pred_pointer[bp + 2 * P];
    float dh = box_pred_pointer[bp + 3 * P];
    dw = _clip_max(dw, 4.1351666);
    dh = _clip_max(dh, 4.1351666);

    // do box transform
    const float ww = x2 - x1 + 1.;
    const float hh = y2 - y1 + 1.;
    const float ctr_x = x1 + 0.5 * ww;
    const float ctr_y = y1 + 0.5 * hh;

    float pred_ctr_x = dx * ww + ctr_x;
    float pred_ctr_y = dy * hh + ctr_y;
    float pred_w = exp(dw) * ww;
    float pred_h = exp(dh) * hh;

    const float xx1 = _clip_max(_clip_min(pred_ctr_x - 0.5 * pred_w, 0.), width_max);
    const float yy1 = _clip_max(_clip_min(pred_ctr_y - 0.5 * pred_h, 0.), height_max);
    const float xx2 = _clip_max(_clip_min(pred_ctr_x + 0.5 * pred_w - 1., 0.), width_max);
    const float yy2 = _clip_max(_clip_min(pred_ctr_y + 0.5 * pred_h - 1., 0.), height_max); 

    pred_w = xx2 - xx1 + 1.;
    pred_h = yy2 - yy1 + 1.;
    pred_ctr_x = xx1 + 0.5 * pred_w;
    pred_ctr_y = yy1 + 0.5 * pred_h;

    rois_raw_pointer[bbp] = xx1;
    rois_raw_pointer[bbp+1] = yy1;
    rois_raw_pointer[bbp+2] = xx2;
    rois_raw_pointer[bbp+3] = yy2;
    rois_raw_pointer[bbp+4] = pred_w * pred_h;
    // not suppressed
    rois_raw_pointer[bbp+5] = (pred_w < 0. || pred_h < 0. || pred_ctr_x > width_max || pred_ctr_y > height_max) ? 1. : 0.;
    index_list.push_back(make_pair(yv_data[r], r));
  }

  sort(index_list.begin(), index_list.end(), _compare_value<float>());
  // then get nms
  int cnt = 0;
  for (int i=0; i<R; i++) {
    const int ind = index_list[i].second;
    const int bp = ind * 6;
    // if not suppressed
    if (rois_raw_pointer[bp+5] == 0.) {
        // leave it untouched
        const float x1A = rois_raw_pointer[bp];
        const float y1A = rois_raw_pointer[bp+1];
        const float x2A = rois_raw_pointer[bp+2];
        const float y2A = rois_raw_pointer[bp+3];
        const float areaA = rois_raw_pointer[bp+4]; 
        // suppress others
        for (int j=i+1; j<R; j++) {
            const int jnd = index_list[j].second;
            const int bbp = jnd * 6;
            const float x1B = rois_raw_pointer[bbp];
            const float y1B = rois_raw_pointer[bbp+1];
            const float x2B = rois_raw_pointer[bbp+2];
            const float y2B = rois_raw_pointer[bbp+3];
            const float areaB = rois_raw_pointer[bbp+4];

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
                rois_raw_pointer[bbp+5] = 1.;
            }
        }
        cnt ++; 
    }
    // enough boxes
    if (cnt == post_top_n_) {
        for (int j=i+1; j<R; j++) {
            const int jnd = index_list[j].second;
            rois_raw_pointer[jnd*6+5] = 1.;
        }
        break;
    }
  }

  // then copy it out
  auto* rois = Output(0);
  auto* roi_probs = Output(1);
  rois->Resize(cnt, 5);
  roi_probs->Resize(cnt, 1);
  float* rois_pointer = rois->mutable_data<float>();
  float* roi_probs_pointer = roi_probs->mutable_data<float>();

  int n = 0;
  for (int i=0; i<R; i++) {
    const int ind = index_list[i].second;
    const int bp = ind * 6;
    if (!rois_raw_pointer[bp+5]) {
      // copy some data
      const int bbp = n * 5;
      rois_pointer[bbp] = static_cast<float>(im_);
      rois_pointer[bbp+1] = rois_raw_pointer[bp];
      rois_pointer[bbp+2] = rois_raw_pointer[bp+1];
      rois_pointer[bbp+3] = rois_raw_pointer[bp+2];
      rois_pointer[bbp+4] = rois_raw_pointer[bp+3];
      roi_probs_pointer[n] = index_list[i].first;
      n++;
    }
  }

  DCHECK_EQ(n, cnt);
  index_list.clear();

  return true;
}

REGISTER_CPU_OPERATOR(GenerateProposalsSingleImage, GenerateProposalsSingleImageOp<float, CPUContext>);

OPERATOR_SCHEMA(GenerateProposalsSingleImage)
    .NumInputs(4)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Return boxes and features given indexes.
)DOC")
    .Arg(
        "pre_top_n",
        "(int) pre-nms top N.")
    .Arg(
        "post_top_n",
        "(int) post-nms top N.")
    .Arg(
        "im",
        "(int) Image Index.")
    .Arg(
        "nms",
        "(float) nms threshold")
    .Arg(
        "stride",
        "(int) stride of the anchors.")
    .Input(
        0,
        "cls_probs",
        "4D feature map input of shape (N, A, H, W).")
    .Input(
        1,
        "box_preds",
        "4D feature map input of shape (N, A * 4, H, W).")
    .Input(
        2,
        "anchors",
        "2D feature map input of shape (A, 4).")
    .Input(
        3,
        "im_info",
        "image information, with (N, 3) shape.")
    .Output(
        0,
        "rois",
        "bounding box information of shape (R, 5): im, x1, y1, x2, y2.")
    .Output(
        1,
        "roi_probs",
        "score information of shape (R, 1).");


SHOULD_NOT_DO_GRADIENT(GenerateProposalsSingleImage);

} // namespace caffe2