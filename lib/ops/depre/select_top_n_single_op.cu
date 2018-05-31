#include <cfloat>

#include "caffe2/core/context_gpu.h"
// #include "caffe2/operators/top_k_heap_selection.cuh"
#include "caffe2/operators/top_k_radix_selection.cuh"
#include "caffe2/utils/math.h"
#include "select_top_n_single_op.h"

namespace caffe2 {

namespace {

template <typename TIndex>
__global__ void SetIndex(const int nthreads, 
                               TIndex* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // just set the index
    output[index] = static_cast<TIndex>(index);
  }
}

// Does not work when K is larger than 512
// template <typename T, int kHeapSize, bool kSelectMax = true>
// void RunHeapSelectionImpl(
//     const T* input,
//     const TIndex outer_size,
//     const TIndex inner_size,
//     const int k,
//     T* values,
//     TIndex* indices,
//     CUDAContext* context) {
//   constexpr int kBlockSize = 256;
//   constexpr int kNumWarps = kBlockSize / kWarpSize;
//   constexpr int smem = kNumWarps * kHeapSize * (sizeof(T) + sizeof(TIndex));
//   constexpr T kInitVal = kSelectMax ? std::numeric_limits<T>::lowest()
//                                     : std::numeric_limits<T>::max();
//   selectRowsViaHeap<T, TIndex, TIndex, kBlockSize, kHeapSize, kSelectMax>
//       <<<outer_size, kBlockSize, smem, context->cuda_stream()>>>(
//           input,
//           values,
//           indices,
//           kInitVal,
//           std::numeric_limits<TIndex>::max(),
//           outer_size,
//           inner_size,
//           k);
// }

// Stupid that it only works when selecting the Bottom K
template <typename T, bool kSelectMax = true>
void RunRadixSelectionImpl(
    const T* input,
    const TIndex outer_size,
    const TIndex inner_size,
    const int k,
    T* values,
    TIndex* indices,
    CUDAContext* context) {
  const int block = std::min(
      math::roundUp(static_cast<int>(inner_size), kWarpSize), CAFFE_CUDA_NUM_THREADS);
  gatherTopK<T, kSelectMax, TIndex>
      <<<outer_size, block, 0, context->cuda_stream()>>>(
          input, inner_size, k, outer_size, values, indices);
}

} // namespace

template<>
bool SelectTopNSingleOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data
  DCHECK_EQ(X.ndim(), 4);
  const int num_images = X.dim32(0);
  DCHECK_LT(im_, num_images);
  const int num_probs = X.dim32(1) * X.dim32(2) * X.dim32(3);
  const float* Xp = X.data<float>() + im_ * num_probs;
  auto* Yi = Output(0);
  auto* Yv = Output(1);

  if (num_probs <= top_n_) {
    // just select everything
    Yi->Resize(num_probs);
    Yv->Resize(num_probs);

    SetIndex<TIndex><<<CAFFE_GET_BLOCKS(num_probs), CAFFE_CUDA_NUM_THREADS,
              0, context_.cuda_stream()>>>(num_probs, 
                                          Yi->mutable_data<TIndex>());
    context_.Copy<float, CUDAContext, CUDAContext>(num_probs, Xp, 
                                                  Yv->mutable_data<float>());

    return true;
  }

  Yi->Resize(top_n_);
  Yv->Resize(top_n_);
  // do the top_k selection thing, heap sort seems not working
  // RunHeapSelectionImpl<float, 1024>(Xp, 
  //                                   1, 
  //                                   num_probs, 
  //                                   top_n_, 
  //                                   Yv->mutable_data<float>(), 
  //                                   Yi->mutable_data<TIndex>(), 
  //                                   &context_);
  RunRadixSelectionImpl<float>(Xp, 
                                1, 
                                num_probs, 
                                top_n_, 
                                Yv->mutable_data<float>(), 
                                Yi->mutable_data<TIndex>(), 
                                &context_);

  return true;
}

REGISTER_CUDA_OPERATOR(SelectTopNSingle,
                       SelectTopNSingleOp<float, CUDAContext>);
} // namespace caffe2