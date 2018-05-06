#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "embed_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

inline __device__
void gpu_atomic_add(const float val, float* address) {
  atomicAdd(address, val);
}

template <typename T>
__global__ void EmbedForward(const int nthreads, 
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
__global__ void EmbedBackward(const int nthreads, const T* input_grad,
                              const int* input2,
                              const int dim, T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int d = index % dim;
    const int i = index / dim;
    const int ii = input2[i];

    gpu_atomic_add(input_grad[index], output_grad + ii * dim + d);
  }
}

} // namespace

template<>
bool EmbedOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data 1
  auto& I = Input(1); // Input data 2
  auto* Y = Output(0); // Output data

  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int R = I.dim32(0);

  Y->Resize(R, C);
  const int output_size = Y->size();

  EmbedForward<float><<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS,
                          0, context_.cuda_stream()>>>(
                          output_size, 
                          X.data<float>(), 
                          I.data<int>(), 
                          C, 
                          Y->mutable_data<float>());

  return true;
}


template<>
bool EmbedGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY  = Input(0);  // Gradient of the output data
  auto& X  = Input(1);  // Input data 1
  auto& I  = Input(2);  // Input data 2
  auto* dX = Output(0); // Gradient of the input data 1

  const int C = X.dim32(1);

  dX->ResizeLike(X);
  const int output_size = dY.size();

  EmbedBackward<float><<<CAFFE_GET_BLOCKS(output_size),
                           CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
                           output_size, 
                           dY.data<float>(),
                           I.data<int>(),
                           C,
                           dX->mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(Embed,
                       EmbedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(EmbedGradient,
                       EmbedGradientOp<float, CUDAContext>);
} // namespace caffe2