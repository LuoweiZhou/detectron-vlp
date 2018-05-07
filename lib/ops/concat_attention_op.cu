#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "concat_attention_op.h"

#include <stdio.h>

namespace caffe2 {

namespace {

template <typename T>
__global__ void ConcatAttentionForward(const int nthreads, const T* bottom_data,
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
__global__ void ConcatAttentionBackward(const int nthreads, const T* input_grad,
    const int channels, const int num_classes, const int pixels,
    const int iter, T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int pidx = index % pixels;
    const int idx = index / pixels;

    const int target_index = (idx * num_classes + iter) * pixels + pidx;
    output_grad[index] = input_grad[target_index];
  } // CUDA_1D_KERNEL_LOOP
} // ConcatAttentionBackward


} // namespace

template<>
bool ConcatAttentionOp<float, CUDAContext>::RunOnDevice() {
  // first calculate the final channel size
  const int num_inputs = InputSize();

  const int N = Input(0).dim32(0);
  const int C = Input(0).dim32(1);
  const int H = Input(0).dim32(2);
  const int W = Input(0).dim32(3);
  const int iter_size = Input(0).size();

  const int CC = C * num_inputs;
  const int pixels = H * W;
  auto* Y = Output(0); 
  Y->Resize(N, CC, H, W);

  for (int iter=0; iter<num_inputs; iter++) {
    auto& X = Input(iter);
    ConcatAttentionForward<float>
        <<<CAFFE_GET_BLOCKS(iter_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(
            iter_size,
            X.data<float>(),
            C, num_inputs, pixels, iter,
            Y->mutable_data<float>());
  }
  return true;
}

template<>
bool ConcatAttentionGradientOp<float, CUDAContext>::RunOnDevice() {
  const int num_inputs = InputSize() - 1;
  auto& dY = Input(0);

  const int N = dY.dim32(0);
  const int CC = dY.dim32(1);
  const int H = dY.dim32(2);
  const int W = dY.dim32(3);

  const int C = CC / num_inputs;
  DCHECK_EQ(CC % num_inputs, 0);
  const int pixels = H * W;

  // Must zero-out dX before accumulating gradients

  for (int iter=0; iter<num_inputs; iter++) {
    auto& X = Input(iter+1);
    auto* dX = Output(iter);
    dX->ResizeLike(X);

    int iter_size = X.size();
    // no need to clean, as it is direct assignment to all the values
    // math::Set<float, CUDAContext>(
    //   dX->size(), 0.f, dX->mutable_data<float>(), &context_);

    ConcatAttentionBackward<float>
        <<<CAFFE_GET_BLOCKS(iter_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(
            iter_size,
            dY.data<float>(),
            C, num_inputs, pixels, iter, 
            dX->mutable_data<float>());
  }
  return true;
}


REGISTER_CUDA_OPERATOR(ConcatAttention,
                       ConcatAttentionOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ConcatAttentionGradient,
                       ConcatAttentionGradientOp<float, CUDAContext>);
} // namespace caffe2