#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "select_fg_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SelectFGForward(const int nthreads, 
                                const T* input, const int* input2, 
                                const int dim,
                                T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int d = index % dim;
    const int i = index / dim;
    const int ii = input2[i];

    output[index] = input[ii * dim + d];
  }
}

template <typename T>
__global__ void SelectFGBackward(const int nthreads, const T* input_grad,
                              const int* input2,
                              const int dim, T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int d = index % dim;
    const int i = index / dim;
    const int ii = input2[i];

    // we make sure it is unique here
    output_grad[ii * dim +d] = input_grad[index];
  }
}

} // namespace

template<>
bool SelectFGOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data 1
  auto& I = Input(1); // Input data 2
  auto* Y = Output(0); // Output data

  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int R = I.dim32(0);

  Y->Resize(R, C);
  const int output_size = Y->size();

  SelectFGForward<float><<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS,
                          0, context_.cuda_stream()>>>(
                          output_size, 
                          X.data<float>(), 
                          I.data<int>(), 
                          C, 
                          Y->mutable_data<float>());

  return true;
}


template<>
bool SelectFGGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY  = Input(0);  // Gradient of the output data
  auto& X  = Input(1);  // Input data 1
  auto& I  = Input(2);  // Input data 2
  auto* dX = Output(0); // Gradient of the input data 1

  const int C = X.dim32(1);

  dX->ResizeLike(X);
  const int output_size = dY.size();

  SelectFGBackward<float><<<CAFFE_GET_BLOCKS(output_size),
                           CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
                           output_size, 
                           dY.data<float>(),
                           I.data<int>(),
                           C,
                           dX->mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(SelectFG,
                       SelectFGOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SelectFGGradient,
                       SelectFGGradientOp<float, CUDAContext>);
} // namespace caffe2