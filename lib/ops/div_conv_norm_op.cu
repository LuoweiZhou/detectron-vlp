#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "div_conv_norm_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void DivConvNormForward(const int nthreads, 
                                const T* input1, const T* input2, 
                                const int channels, const int pixels,
                                T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int idx = index % pixels;
    const int b = (index / pixels) / channels;
    const float divider = input2[b * pixels + idx];
    // only do safe division
    if (divider > 0.) {
      output[index] = input1[index] / divider;
    } else {
      output[index] = 0.;
    }
  }
}

template <typename T>
__global__ void DivConvNormBackward(const int nthreads, 
    const T* input_grad, const T* input1, const T* input2,
    const int channels, const int pixels, T* output_grad1) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int idx = index % pixels;
    const int b = (index / pixels) / channels;
    const float divider = input2[b * pixels + idx];
    // only deal with input 1
    if (divider > 0.) {
      output_grad1[index] = input_grad[index] / divider;
    } else {
      output_grad1[index] = 0.;
    }
  }
}

} // namespace

template<>
bool DivConvNormOp<float, CUDAContext>::RunOnDevice() {
  auto& X1 = Input(0); // Input data 1
  auto& X2 = Input(1); // Input data 2
  auto* Y = Output(0); // Output data, summation of the two

  const int N = X1.dim32(0);
  const int C = X1.dim32(1);
  const int H = X1.dim32(2);
  const int W = X1.dim32(3);

  const int A = X2.dim32(1);
  DCHECK_EQ(N, X2.dim32(0));
  DCHECK_EQ(C % A, 0);
  DCHECK_EQ(H, X2.dim32(2));
  DCHECK_EQ(W, X2.dim32(3));

  const int pixels = H * W;
  const int X = C / A;

  // N, C, H, W
  Y->Resize(N, C, H, W);
  const int output_size = Y->size();
  DivConvNormForward<float><<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS,
                          0, context_.cuda_stream()>>>(
                          output_size, X1.data<float>(), X2.data<float>(), 
                          X, pixels, Y->mutable_data<float>());

  return true;
}


template<>
bool DivConvNormGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY  = Input(0);  // Gradient of the output data
  auto& X1  = Input(1);  // Input data 1
  auto& X2  = Input(2);  // Input data 2
  auto* dX1 = Output(0); // Gradient of the input data 1

  const int C = X1.dim32(1);
  const int H = X1.dim32(2);
  const int W = X1.dim32(3);

  const int A = X2.dim32(1);
  const int X = C / A;

  const int pixels = H * W;
  const int output_size = dY.size();

  dX1->ResizeLike(X1);

  DivConvNormBackward<float><<<CAFFE_GET_BLOCKS(output_size),
                           CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
                                 output_size, 
                                 dY.data<float>(),
                                 X1.data<float>(),
                                 X2.data<float>(),
                                 X,
                                 pixels,
                                 dX1->mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(DivConvNorm,
                       DivConvNormOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(DivConvNormGradient,
                       DivConvNormGradientOp<float, CUDAContext>);
} // namespace caffe2