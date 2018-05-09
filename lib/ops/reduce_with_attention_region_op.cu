#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "reduce_with_attention_region_op.h"

#include <stdio.h>

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__
float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void ReduceWithAttentionRegionForward(const int nthreads, 
                              const T* bottom_data, const T* attention_data,
                              const int num_inputs,
                              const int A, const int X,
                              const int iter, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int idx = index / X;
    const int a = idx % A;
    const int n = idx / A;

    const int target_index = (n * num_inputs + iter) * A + a;
    top_data[index] += bottom_data[index] * attention_data[target_index];
  }
}

template <typename T>
__global__ void ReduceWithAttentionRegionBackward(const int nthreads, const T* input_grad,
                            const T* bottom_data, const int num_inputs,
                            const int A, const int X,
                            const int iter, T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int idx = index / X;
    const int a = idx % A;
    const int n = idx / A;

    const int target_index = (n * num_inputs + iter) * A + a;
    if (X == 1)
      output_grad[target_index] = input_grad[index] * bottom_data[index];
    else
      gpu_atomic_add(input_grad[index] * bottom_data[index], 
                    output_grad + target_index);
  } // CUDA_1D_KERNEL_LOOP
} // ReduceWithAttentionRegionBackward


} // namespace

template<>
bool ReduceWithAttentionRegionOp<float, CUDAContext>::RunOnDevice() {
  // first calculate the final channel size
  const int num_inputs = InputSize() - 1;
  DCHECK_EQ(num_inputs, iter_);
  auto& Attention = Input(0);
  auto* Y = Output(0); 

  const int N = Attention.dim32(0);
  const int C = Attention.dim32(1);
  const int A = C / iter_;
  DCHECK_EQ(C % iter_, 0);

  const int D = Input(1).dim32(1);
  const int X = D / A;
  DCHECK_EQ(D % A, 0);

  // resize as the first input, or any input afterwards
  Y->ResizeLike(Input(1));
  const int output_size = Y->size();
  math::Set<float, CUDAContext>(
       output_size, 0.f, Y->mutable_data<float>(), &context_);

  for (int iter=0; iter<num_inputs; iter++) {
    auto& Xstar = Input(iter+1);
    ReduceWithAttentionRegionForward<float>
        <<<CAFFE_GET_BLOCKS(output_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(
            output_size,
            Xstar.data<float>(),
            Attention.data<float>(),
            num_inputs, A, X, iter,
            Y->mutable_data<float>()
        );
  }
  return true;
}

template<>
bool ReduceWithAttentionRegionGradientOp<float, CUDAContext>::RunOnDevice() {
  const int num_inputs = InputSize() - 2;
  DCHECK_EQ(num_inputs, iter_);
  auto& dY = Input(0);
  auto& Attention = Input(1);

  const int N = Attention.dim32(0);
  const int C = Attention.dim32(1);
  const int A = C / iter_;

  const int D = Input(2).dim32(1);
  const int X = D / A;

  auto* dA = Output(0);
  dA->ResizeLike(Attention);
  const int output_size = dY.size();
  // Must zero-out dA before accumulating gradients
  math::Set<float, CUDAContext>(
       dA->size(), 0.f, dA->mutable_data<float>(), &context_);

  for (int iter=0; iter<num_inputs; iter++) {
    auto& Xstar = Input(iter+2);
    ReduceWithAttentionRegionBackward<float>
        <<<CAFFE_GET_BLOCKS(output_size),
           CAFFE_CUDA_NUM_THREADS, 0,
        context_.cuda_stream()>>>(
            output_size,
            dY.data<float>(),
            Xstar.data<float>(),
            num_inputs, A, X, iter,
            dA->mutable_data<float>()
        );
  }
  return true;
}


REGISTER_CUDA_OPERATOR(ReduceWithAttentionRegion,
                       ReduceWithAttentionRegionOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReduceWithAttentionRegionGradient,
                       ReduceWithAttentionRegionGradientOp<float, CUDAContext>);
} // namespace caffe2