#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "get_logits_op.h"

namespace caffe2 {

namespace {

__global__ void TryAtThisLevel(const int nthreads,
                               const float* frois_pointer,
                               const float* cls_preds_pointer,
                               const int num_classes,
                               const int height,
                               const int width,
                               const int pixel,
                               const int A,
                               const int level,
                               float* output_feats) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int bp = i * 6;
    const int lvl = static_cast<int>(frois_pointer[bp+2]);
    if (lvl == level) {
      const int im = static_cast<int>(frois_pointer[bp]);
      const int a = static_cast<int>(frois_pointer[bp+3]);
      const int h = static_cast<int>(frois_pointer[bp+4]);
      const int w = static_cast<int>(frois_pointer[bp+5]);
      
      const int fp = i * num_classes;
      const int ffp = ((im * A + a) * num_classes * height + h) * width + w;

      for (int j=0, jj=0; j<num_classes; j++, jj+=pixel) {
        output_feats[fp + j] = cls_preds_pointer[ffp + jj];
      }
    }
  }
}

__global__ void TrySingleAtThisLevel(const int nthreads,
                               const float* frois_pointer,
                               const float* cls_preds_pointer,
                               const int num_classes,
                               const int height,
                               const int width,
                               const int pixel,
                               const int A,
                               const int level,
                               float* output_feats) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int bp = i * 6;
    const int lvl = static_cast<int>(frois_pointer[bp+2]);
    if (lvl == level) {
      const int im = static_cast<int>(frois_pointer[bp]);
      const int c = static_cast<int>(frois_pointer[bp+1]);
      const int a = static_cast<int>(frois_pointer[bp+3]);
      const int h = static_cast<int>(frois_pointer[bp+4]);
      const int w = static_cast<int>(frois_pointer[bp+5]);
      
      const int fp = i * num_classes;
      const int ffp = (((im * A + a) * num_classes + c) * height + h) * width + w;

      output_feats[fp + c] = cls_preds_pointer[ffp];
    }
  }
}

} // namespace

template<>
bool GetLogitsOp<float, CUDAContext>::RunOnDevice() {
  auto& frois = Input(0);
  const int num_total = frois.dim32(0);
  DCHECK_EQ(frois.dim32(1), 6);
  const float* frois_pointer = frois.data<float>();
  const int num_inputs = InputSize();
  // also including the extra frois
  DCHECK_EQ(num_inputs, k_max_ - k_min_ + 2);
  auto* output_feats = Output(0);
  output_feats->Resize(num_total, num_feats_);
  float* output_feats_pointer = output_feats->mutable_data<float>();
  if (!all_dim_)
    math::Set<float, CUDAContext>(num_total * num_feats_, 0.f, output_feats_pointer, &context_);

  if (num_total > 0) {
    int lvl = k_min_;
    for (int i=1; i<num_inputs; i++) {
      auto& cls_preds = Input(i);
      const float* cls_preds_pointer = cls_preds.data<float>();

      const int N = cls_preds.dim32(0);
      const int C = cls_preds.dim32(1) / A_;
      if (i == 1)
        DCHECK_EQ(C, num_feats_);
      const int H = cls_preds.dim32(2);
      const int W = cls_preds.dim32(3);
      const int P = H * W;

      if (all_dim_) {
        TryAtThisLevel<<<CAFFE_GET_BLOCKS(num_total), CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(num_total,
                                                        frois_pointer,
                                                        cls_preds_pointer,
                                                        C, H, W,
                                                        P, A_, lvl,
                                                        output_feats_pointer);
      } else {
        TrySingleAtThisLevel<<<CAFFE_GET_BLOCKS(num_total), CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(num_total,
                                                        frois_pointer,
                                                        cls_preds_pointer,
                                                        C, H, W,
                                                        P, A_, lvl,
                                                        output_feats_pointer);
      }

      lvl ++;
    }
  }

  return true;
}

REGISTER_CUDA_OPERATOR(GetLogits,
                       GetLogitsOp<float, CUDAContext>);
} // namespace caffe2