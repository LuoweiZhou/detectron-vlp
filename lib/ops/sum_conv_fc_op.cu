#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "sum_conv_fc_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SumConvFCForward(const int nthreads, 
                                const T* input1, const T* input2, 
                                const int pixels,
                                T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int idx = index / pixels;
    output[index] = input1[index] + input2[idx];
  }
}

template <typename T>
__global__ void SumConvFCBackward(const int nthreads, const T* input_grad,
    const int pixels, T* output_grad1, T* output_grad2) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // first for X1
    output_grad1[index] = input_grad[index];

    // only doing this for the first element, not sure if it is
    // the most efficient way though
    if (index % pixels == 0) {
      const int idx = index / pixels;

      const int base_start = idx * pixels;
      const int base_end = base_start + pixels;

      // just summing things up
      T grad = 0.;
      for (int i=base_start; i<base_end; i++) {
        grad += input_grad[i];
      }
      output_grad2[idx] = grad;
    }
  }
}

} // namespace

template<>
bool SumConvFCOp<float, CUDAContext>::RunOnDevice() {
  auto& X1 = Input(0);  // Input data 1
  auto& X2 = Input(1); // Input data 2
  auto* Y = Output(0); // Output data, summation of the two

  const int N = X1.dim32(0);
  const int C = X1.dim32(1);
  const int H = X1.dim32(2);
  const int W = X1.dim32(3);

  DCHECK_EQ(N, X2.dim32(0));
  DCHECK_EQ(C, X2.dim32(1));
  DCHECK_EQ(X2.ndim(),2);

  const int pixels = H * W;

  // N, C, H, W
  Y->Resize(N, C, H, W);
  const int output_size = Y->size();
  SumConvFCForward<float><<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS,
                          0, context_.cuda_stream()>>>(
                          output_size, X1.data<float>(), X2.data<float>(), 
                          pixels, Y->mutable_data<float>());

  return true;
}


template<>
bool SumConvFCGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY  = Input(0);  // Gradient of the output data
  auto& X1  = Input(1);  // Input data 1
  auto& X2  = Input(2);  // Input data 2
  auto* dX1 = Output(0); // Gradient of the input data 1
  auto* dX2 = Output(1); // Gradient of the input data 2

  const int H = X1.dim32(2);
  const int W = X1.dim32(3);

  const int pixels = H * W;
  const int output_size = dY.size();

  dX1->ResizeLike(X1);
  dX2->ResizeLike(X2);

  SumConvFCBackward<float><<<CAFFE_GET_BLOCKS(output_size),
                           CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
                           output_size, dY.data<float>(),
                           pixels,
                           dX1->mutable_data<float>(),
                           dX2->mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(SumConvFC,
                       SumConvFCOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SumConvFCGradient,
                       SumConvFCGradientOp<float, CUDAContext>);
} // namespace caffe2