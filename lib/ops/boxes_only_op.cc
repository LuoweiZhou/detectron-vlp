#include "boxes_only_op.h"

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

}

template<>
bool BoxesOnlyOp<float, CPUContext>::RunOnDevice() {
  auto& box_preds = Input(0);
  auto& anchors = Input(1);
  auto& YI = Input(2);
  auto& YV = Input(3);
  auto& im_info = Input(4);

  // get some sizes
  const int N = box_preds.dim32(0);
  DCHECK_EQ(box_preds.dim32(1) / A_, 4);
  const int height = box_preds.dim32(2);
  const int width = box_preds.dim32(3);
  const int pixel = height * width;
  const int offset_box = im_ * A_ * 4 * pixel;
  const int offset_info = im_ * 3;
  const float stride = pow(2., lvl_);
  const int R = YI.dim32(0);

  auto* boxes = Output(0);
  boxes->Resize(R, 12);
  auto* stats = Output(1);
  stats->Resize(num_cls_);
  int* stats_pointer = stats->mutable_data<int>();
  _zeros(num_cls_, stats_pointer);

  const float* box_preds_pointer = box_preds.data<float>() + offset_box;
  const float* anchors_pointer = anchors.data<float>();
  const TIndex* YI_pointer = YI.data<TIndex>();
  const float* YD_pointer = YV.data<float>();
  const float* im_info_pointer = im_info.data<float>() + offset_info;
  float* boxes_pointer = boxes->mutable_data<float>();

  for (int r=0, bbp=0; r<R; r++, bbp+=12) {
    const int index = static_cast<int>(YI_pointer[r]);
    int ind = index;
    const int w = ind % width;
    ind /= width;
    const int h = ind % height;
    ind /= height;
    const int c = ind % num_cls_;
    const int a = ind / num_cls_;
    const int ap = a * 4;
    const int bp = (a * 4 * height + h) * width + w;

    const float x = w * stride;
    const float y = h * stride;

    const float x1 = x + anchors_pointer[ap];
    const float y1 = y + anchors_pointer[ap+1];
    const float x2 = x + anchors_pointer[ap+2];
    const float y2 = y + anchors_pointer[ap+3];

    const float dx = box_preds_pointer[bp];
    const float dy = box_preds_pointer[bp + pixel];
    float dw = box_preds_pointer[bp + 2 * pixel];
    float dh = box_preds_pointer[bp + 3 * pixel];
    dw = _clip_max(dw, 4.1351666);
    dh = _clip_max(dh, 4.1351666);

    // do box transform
    const float ww = x2 - x1 + 1.;
    const float hh = y2 - y1 + 1.;
    const float ctr_x = x1 + 0.5 * ww;
    const float ctr_y = y1 + 0.5 * hh;

    const float pred_ctr_x = dx * ww + ctr_x;
    const float pred_ctr_y = dy * hh + ctr_y;
    float pred_w = exp(dw) * ww;
    float pred_h = exp(dh) * hh;

    const float height_max = im_info_pointer[0] - 1.;
    const float width_max = im_info_pointer[1] - 1.;

    const float xx1 = _clip_max(_clip_min(pred_ctr_x - 0.5 * pred_w, 0.), width_max);
    const float yy1 = _clip_max(_clip_min(pred_ctr_y - 0.5 * pred_h, 0.), height_max);
    const float xx2 = _clip_max(_clip_min(pred_ctr_x + 0.5 * pred_w - 1., 0.), width_max);
    const float yy2 = _clip_max(_clip_min(pred_ctr_y + 0.5 * pred_h - 1., 0.), height_max); 

    pred_w = xx2 - xx1 + 1.;
    pred_h = yy2 - yy1 + 1.;

    boxes_pointer[bbp] = xx1;
    boxes_pointer[bbp + 1] = yy1;
    boxes_pointer[bbp + 2] = xx2;
    boxes_pointer[bbp + 3] = yy2;
    // because of the bug, we need to first negate the cls_prob
    boxes_pointer[bbp + 4] = YD_pointer[r];
    boxes_pointer[bbp + 5] = static_cast<float>(c);
    // valid: 1., invalid: 0.
    const float valid = (pred_w < 1. || pred_h < 1.) ? 0. : 1.;
    boxes_pointer[bbp + 6] = valid;
    // for getting the features
    boxes_pointer[bbp + 7] = static_cast<float>(im_);
    boxes_pointer[bbp + 8] = static_cast<float>(lvl_);
    boxes_pointer[bbp + 9] = static_cast<float>(a);
    boxes_pointer[bbp + 10] = static_cast<float>(h);
    boxes_pointer[bbp + 11] = static_cast<float>(w);

    if (valid) {
      stats_pointer[c] ++;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(BoxesOnly, BoxesOnlyOp<float, CPUContext>);

OPERATOR_SCHEMA(BoxesOnly)
    .NumInputs(5)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Return boxes and features given indexes.
)DOC")
    .Arg(
        "A",
        "(int) Product of scale and aspect ratio per location.")
    .Arg(
        "im",
        "(int) Image Index.")
    .Arg(
        "level",
        "(int) Level of the current detection.")
    .Arg(
        "num_cls",
        "(int) Number of classes.")
    .Input(
        0,
        "box_preds",
        "4D feature map input of shape (N, A * 4, H, W).")
    .Input(
        1,
        "anchors",
        "2D feature map input of shape (A, 4).")
    .Input(
        2,
        "YI",
        "location index of shape (R).")
    .Input(
        3,
        "YV",
        "location value of shape (R).")
    .Input(
        4,
        "im_info",
        "image information.")
    .Output(
        0,
        "boxes",
        "bounding box information of shape (R, 12): x1, y1, x2, y2, score, c, valid, im, lvl, a, h, w.")
    .Output(
        1,
        "stats",
        "Number of bounding boxes assigned to each class. (num_cls)");


SHOULD_NOT_DO_GRADIENT(BoxesOnly);

} // namespace caffe2