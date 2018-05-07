#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "concat_plus_attention_op.h"

#include <stdio.h>

namespace caffe2 {

namespace {

template <typename T>
__global__ void SetZeroForward(const int nthreads, 
    const int channels, const int num_classes, const int pixels, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int pidx = index % pixels;
    const int idx = index / pixels;

    const int target_index = idx * num_classes * pixels + pidx;
    top_data[target_index] = 0.;
  }
}

template <typename T>
__global__ void ConcatPlusAttentionForward(const int nthreads, const T* bottom_data,
    const int channels, const int num_classes, const int pixels,
    const int iter, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int pidx = index % pixels;
    const int idx = index / pixels;

    const int target_index = (idx * num_classes + iter) * pixels + pidx;
    top_data[target_index] = bottom_data[index];
  }
}

template <typename T>
__global__ void ConcatPlusAttentionBackward(const int nthreads, const T* input_grad,
    const int channels, const int num_classes, const int pixels,
    const int iter, T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int pidx = index % pixels;
    const int idx = index / pixels;

    const int target_index = (idx * num_classes + iter) * pixels + pidx;
    output_grad[index] = input_grad[target_index];
  } // CUDA_1D_KERNEL_LOOP
} // ConcatPlusAttentionBackward


} // namespace

template<>
bool ConcatPlusAttentionOp<float, CUDAContext>::RunOnDevice() {
  // first calculate the final channel size
  const int num_inputs = InputSize();

  const int N = Input(0).dim32(0);
  const int C = Input(0).dim32(1);
  const int H = Input(0).dim32(2);
  const int W = Input(0).dim32(3);
  // assuming all the blobs are of the same size
  const int iter_size = Input(0).size();
  const int num_classes = num_inputs + 1;
  const int CC = C * num_classes;
  const int pixels = H * W;
  auto* Y = Output(0); 
  Y->Resize(N, CC, H, W);
  float* Yp = Y->mutable_data<float>();

  // set the first class to be zero
  SetZeroForward<float>
        <<<CAFFE_GET_BLOCKS(iter_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(iter_size, C, num_classes, pixels, Yp);

  for (int iter=0; iter<num_inputs; iter++) {
    auto& X = Input(iter);
    ConcatPlusAttentionForward<float>
        <<<CAFFE_GET_BLOCKS(iter_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(
            iter_size,
            X.data<float>(),
            C, num_classes, pixels, iter+1, Yp);
  }
  return true;
}

template<>
bool ConcatPlusAttentionGradientOp<float, CUDAContext>::RunOnDevice() {
  const int num_classes = InputSize();
  auto& dY = Input(0);

  const int N = dY.dim32(0);
  const int CC = dY.dim32(1);
  const int H = dY.dim32(2);
  const int W = dY.dim32(3);

  const int C = CC / num_classes;
  const int pixels = H * W;
  const float* Yp = dY.data<float>();

  // Only back propagate from the second class on
  for (int iter=1; iter<num_classes; iter++) {
    auto& X = Input(iter);
    int iter_size = X.size();
    auto* dX = Output(iter-1);
    dX->ResizeLike(X);

    ConcatPlusAttentionBackward<float>
        <<<CAFFE_GET_BLOCKS(iter_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(
            iter_size,
            Yp, C, num_classes, pixels, iter, 
            dX->mutable_data<float>());
  }
  return true;
}


REGISTER_CUDA_OPERATOR(ConcatPlusAttention,
                       ConcatPlusAttentionOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ConcatPlusAttentionGradient,
                       ConcatPlusAttentionGradientOp<float, CUDAContext>);
} // namespace caffe2