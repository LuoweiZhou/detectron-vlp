#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "boxes_and_feats_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__
int gpu_atomic_add(const int val, int* address) {
  return atomicAdd(address, val);
}

inline __device__ float _clip_min(const float f, const float m) {
    return (f > m) ? (f) : (m);
}

inline __device__ float _clip_max(const float f, const float m) {
    return (f < m) ? (f) : (m);
}

template <typename T>
__global__ void BoxesAndFeatsForward(const int nthreads,
                                    const T* cls_pred,
                                    const T* box_pred,
                                    const T* anchor,
                                    const TIndex* YI,
                                    const T* YV, 
                                    const T* info,
                                    const T stride,
                                    const int A,
                                    const int num_classes,
                                    const int height,
                                    const int width,
                                    const int pixel,
                                    T* boxes_pointer,
                                    T* feats_pointer,
                                    int* stats_pointer) {
  CUDA_1D_KERNEL_LOOP(r, nthreads) {
    const T cls_prob_val = -YV[r];
    const int index = static_cast<int>(YI[r]);
    int ind = index;
    const int w = ind % width;
    ind /= width;
    const int h = ind % height;
    ind /= height;
    const int c = ind % num_classes;
    const int a = ind / num_classes;
    const int ap = a * 4;
    const int bp = (a * 4 * height + h) * width + w;

    const T x = w * stride;
    const T y = h * stride;

    const T x1 = x + anchor[ap];
    const T y1 = y + anchor[ap+1];
    const T x2 = x + anchor[ap+2];
    const T y2 = y + anchor[ap+3];

    const T dx = box_pred[bp];
    const T dy = box_pred[bp + pixel];
    T dw = box_pred[bp + 2 * pixel];
    T dh = box_pred[bp + 3 * pixel];
    dw = _clip_max(dw, 4.1351666);
    dh = _clip_max(dh, 4.1351666);

    // do box transform
    const T ww = x2 - x1 + 1.;
    const T hh = y2 - y1 + 1.;
    const T ctr_x = x1 + 0.5 * ww;
    const T ctr_y = y1 + 0.5 * hh;

    const T pred_ctr_x = dx * ww + ctr_x;
    const T pred_ctr_y = dy * hh + ctr_y;
    T pred_w = exp(dw) * ww;
    T pred_h = exp(dh) * hh;

    const T height_max = info[0] - 1.;
    const T width_max = info[1] - 1.;

    const T xx1 = _clip_max(_clip_min(pred_ctr_x - 0.5 * pred_w, 0.), width_max);
    const T yy1 = _clip_max(_clip_min(pred_ctr_y - 0.5 * pred_h, 0.), height_max);
    const T xx2 = _clip_max(_clip_min(pred_ctr_x + 0.5 * pred_w - 1., 0.), width_max);
    const T yy2 = _clip_max(_clip_min(pred_ctr_y + 0.5 * pred_h - 1., 0.), height_max); 

    pred_w = xx2 - xx1 + 1.;
    pred_h = yy2 - yy1 + 1.;

    // then dump the data
    const int bbp = r * 7;
    boxes_pointer[bbp] = xx1;
    boxes_pointer[bbp + 1] = yy1;
    boxes_pointer[bbp + 2] = xx2;
    boxes_pointer[bbp + 3] = yy2;
    // because of the bug, we need to first negate the cls_prob
    boxes_pointer[bbp + 4] = cls_prob_val;
    boxes_pointer[bbp + 5] = static_cast<float>(c);
    // valid: 1., invalid: 0.
    const T valid = (pred_w < 1. || pred_h < 1.) ? 0. : 1.;
    boxes_pointer[bbp + 6] = valid;

    // accumulate stats
    if (valid) {
      gpu_atomic_add(1, stats_pointer + c);
    }

    const int fp = r * num_classes;
    const int ffp = (a * num_classes * height + h) * width + w;
    // for loop to do the copy
    for (int j=0, jj=0; j<num_classes; j++, jj+=pixel) {
      feats_pointer[fp + j] = cls_pred[ffp + jj];
    }
  }
}

} // namespace

template<>
bool BoxesAndFeatsOp<float, CUDAContext>::RunOnDevice() {
  auto& cls_preds = Input(0); 
  auto& box_preds = Input(1);
  auto& anchors = Input(2);
  auto& YI = Input(3);
  auto& YV = Input(4);
  auto& im_info = Input(5);

  // get some sizes
  const int N = cls_preds.dim32(0);
  const int num_cls = cls_preds.dim32(1) / A_;
  DCHECK_EQ(box_preds.dim32(1) / A_, 4);
  const int H = cls_preds.dim32(2);
  const int W = cls_preds.dim32(3);
  const int pixel = H * W;
  const int num_probs = A_ * num_cls * pixel;
  const int offset_cls = im_ * num_probs;
  const int offset_box = im_ * A_ * 4 * pixel;
  const int offset_info = im_ * 3;
  const float stride = pow(2., lvl_);
  const int R = YI.dim32(0);

  auto* boxes = Output(0);
  boxes->Resize(R, 7);
  auto* feats = Output(1);
  feats->Resize(R, num_cls);
  auto* stats = Output(2);
  stats->Resize(num_cls);
  int* stats_pointer = stats->mutable_data<int>();
  math::Set<int, CUDAContext>(num_cls, 0, stats_pointer, &context_);

  BoxesAndFeatsForward<float><<<CAFFE_GET_BLOCKS(R), CAFFE_CUDA_NUM_THREADS,
                    0, context_.cuda_stream()>>>(R, 
                                                 cls_preds.data<float>() + offset_cls,
                                                 box_preds.data<float>() + offset_box,
                                                 anchors.data<float>(),
                                                 YI.data<TIndex>(),
                                                 YV.data<float>(),
                                                 im_info.data<float>() + offset_info,
                                                 stride, A_, num_cls, H, W, pixel,
                                                 boxes->mutable_data<float>(),
                                                 feats->mutable_data<float>(),
                                                 stats_pointer);

  return true;
}

REGISTER_CUDA_OPERATOR(BoxesAndFeats,
                       BoxesAndFeatsOp<float, CUDAContext>);
} // namespace caffe2