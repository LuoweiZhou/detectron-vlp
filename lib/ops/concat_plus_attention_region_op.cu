#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "concat_plus_attention_region_op.h"

#include <stdio.h>

namespace caffe2 {

namespace {

template <typename T>
__global__ void SetZeroForward(const int nthreads, 
    const int channels, const int num_classes, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    top_data[index * num_classes] = 0.;
  }
}

template <typename T>
__global__ void ConcatPlusAttentionRegionForward(const int nthreads, const T* bottom_data,
    const int channels, const int num_classes, 
    const int iter, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int target_index = index * num_classes + iter;
    top_data[target_index] = bottom_data[index];
  }
}

template <typename T>
__global__ void ConcatPlusAttentionRegionBackward(const int nthreads, const T* input_grad,
    const int channels, const int num_classes,
    const int iter, T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int target_index = index * num_classes + iter;
    output_grad[index] = input_grad[target_index];
  } // CUDA_1D_KERNEL_LOOP
} // ConcatPlusAttentionRegionBackward


} // namespace

template<>
bool ConcatPlusAttentionRegionOp<float, CUDAContext>::RunOnDevice() {
  // first calculate the final channel size
  const int num_inputs = InputSize();

  const int N = Input(0).dim32(0);
  const int C = Input(0).dim32(1);

  // assuming all the blobs are of the same size
  const int iter_size = Input(0).size();
  const int num_classes = num_inputs + 1;
  const int CC = C * num_classes;

  auto* Y = Output(0); 
  Y->Resize(N, CC);
  float* Yp = Y->mutable_data<float>();

  // set the first class to be zero
  SetZeroForward<float>
        <<<CAFFE_GET_BLOCKS(iter_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(iter_size, C, num_classes, Yp);

  for (int iter=0; iter<num_inputs; iter++) {
    auto& X = Input(iter);
    ConcatPlusAttentionRegionForward<float>
        <<<CAFFE_GET_BLOCKS(iter_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(
            iter_size,
            X.data<float>(),
            C, num_classes, iter+1, Yp);
  }
  return true;
}

template<>
bool ConcatPlusAttentionRegionGradientOp<float, CUDAContext>::RunOnDevice() {
  const int num_classes = InputSize();
  auto& dY = Input(0);

  const int N = dY.dim32(0);
  const int CC = dY.dim32(1);

  const int C = CC / num_classes;
  const float* Yp = dY.data<float>();

  // Only back propagate from the second class on
  for (int iter=1; iter<num_classes; iter++) {
    auto& X = Input(iter);
    int iter_size = X.size();
    auto* dX = Output(iter-1);
    dX->ResizeLike(X);

    ConcatPlusAttentionRegionBackward<float>
        <<<CAFFE_GET_BLOCKS(iter_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(
            iter_size,
            Yp, C, num_classes, iter, 
            dX->mutable_data<float>());
  }
  return true;
}


REGISTER_CUDA_OPERATOR(ConcatPlusAttentionRegion,
                       ConcatPlusAttentionRegionOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ConcatPlusAttentionRegionGradient,
                       ConcatPlusAttentionRegionGradientOp<float, CUDAContext>);
} // namespace caffe2