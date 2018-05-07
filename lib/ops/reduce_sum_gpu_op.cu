#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "reduce_sum_gpu_op.h"

namespace caffe2 {

namespace {

__global__ void ReduceSumGPUForward(const int nthreads,
                                      const int* input_stats,
                                      int* stats) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    stats[i] += input_stats[i];
  }
}

} // namespace

template<>
bool ReduceSumGPUOp<int, CUDAContext>::RunOnDevice() {
  const int num_inputs = InputSize();  
  auto* Y = Output(0);
  int num_feat;
  int* stats;
  for (int i=0; i<num_inputs; i++) {
    auto& X = Input(i);
    if (i==0) {
      num_feat = X.dim32(0);
      Y->Resize(num_feat);
      stats = Y->mutable_data<int>();

      context_.Copy<int, CUDAContext, CUDAContext>(num_feat, 
                                                   X.data<int>(), 
                                                   stats);

    } else {
      ReduceSumGPUForward<<<CAFFE_GET_BLOCKS(num_feat), CAFFE_CUDA_NUM_THREADS,
                               0, context_.cuda_stream()>>>(num_feat,
                                                            X.data<int>(),
                                                            stats);
    }
  }

  return true;
}

REGISTER_CUDA_OPERATOR(ReduceSumGPU,
                       ReduceSumGPUOp<int, CUDAContext>);
} // namespace caffe2