#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "add_spatial_softmax_op.h"

namespace caffe2 {

namespace {

__global__ void AddSpatialSoftmaxKernel(const int N, const int A, 
                                        const int pixels,
                                        const float* Xdata, float* Pdata, 
                                        const int num_classes) {
  // Loop throuh labels (N x A x H x W)
  CUDA_1D_KERNEL_LOOP(index, N * A * pixels) {
    const int DP = num_classes * A;
    const int D = DP - A;
    const int num_dim = num_classes - 1;
    const int p = index % pixels;
    const int a = (index / pixels) % A;
    const int i = index / (pixels * A);

    float max_val = 0;
    const int start_bottom = a * num_dim;
    const int end_bottom = start_bottom + num_dim;
    const int start_top = a * num_classes;
    const int end_top= start_top + num_classes;

    // Subtract max on each cell for numerical reasons
    for(int c = start_bottom; c < end_bottom; ++c) {
      int idx = (i * D + c) * pixels + p;
      max_val = max(max_val, Xdata[idx]);
    }
    
    // Exponentiate
    float expsum = exp(-max_val);
    int tc = start_top;
    Pdata[(i * DP + tc) * pixels + p] = expsum;
    for(int c = start_bottom; c < end_bottom; ++c) {
      int idx = (i * D + c) * pixels + p;
      float expx = exp(Xdata[idx] - max_val);
      tc ++;
      int tidx = (i * DP + tc) * pixels + p;
      Pdata[tidx] = expx;
      expsum += expx;
    }

    // Normalize
    for(int tc = start_top; tc < end_top; ++tc) {
      int tidx = (i * DP + tc) * pixels + p;
      Pdata[tidx] /= expsum;
    }
  }
}

__global__ void DeCopyKernel(const int num, const int A, 
                          const int pixels,
                          const float* dYdata,
                          float* dXdata, 
                          const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, num) {
    int idx = index;
    const int num_dim = num_classes - 1;
    const int p = idx % pixels;
    idx /= pixels;
    const int c = idx % num_dim;
    idx /= num_dim;

    const int target_index = (idx * num_classes + c + 1) * pixels + p;
    dXdata[index] = dYdata[target_index];
  }
}

__global__ void SumProbsKernel(const int N, const int A, const int pixels, 
                              const float* Ydata, const float* dYdata,
                              float* sum_probs_data, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, N * A * pixels) {
    int DP = num_classes * A;
    int p = index % pixels;
    int a = (index / pixels) % A;
    int i = index / (pixels * A);

    const int start_top = a * num_classes;
    const int end_top= start_top + num_classes;

    float sum = 0.;
    for(int c = start_top; c < end_top; ++c) {
      int tidx = (i * DP + c) * pixels + p;
      sum += Ydata[tidx] * dYdata[tidx];
    }

    sum_probs_data[index] = sum;
  }
}

__global__ void SubSumKernel(const int N, const int A, const int pixels,
                            const float* sum_probs_data, float* dXdata, 
                            const int num_dim) {
  CUDA_1D_KERNEL_LOOP(index, N * A * num_dim * pixels) {
    const int p = index % pixels;
    const int idx = index / (pixels * num_dim);

    const int sidx = idx * pixels + p;
    dXdata[index] = dXdata[index] - sum_probs_data[sidx];
  }
}

__global__ void DeMulKernel(const int num, const int A, 
                          const int pixels,
                          const float* Ydata,
                          float* dXdata, 
                          const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, num) {
    int idx = index;
    const int num_dim = num_classes - 1;
    const int p = idx % pixels;
    idx /= pixels;
    const int c = idx % num_dim;
    idx /= num_dim;

    const int target_index = (idx * num_classes + c + 1) * pixels + p;
    dXdata[index] *= Ydata[target_index];
  }
}

} // namespace


template <>
bool AddSpatialSoftmaxOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Logits
  auto* P = Output(0); // Probabilities from softmax

  const int N = X.dim32(0);
  const int D = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const int pixels = H * W;
  const int A = D / (num_classes_ - 1);
  // additional dimension
  const int DP = D + A;

  P->Resize(N, DP, H, W);
  DCHECK_EQ(X.ndim(), 4);

  const float* Xdata = X.data<float>();
  float* Pdata = P->mutable_data<float>();

  // Softmax for each x,y location
  AddSpatialSoftmaxKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                         0, context_.cuda_stream()>>>(
                            N, A, pixels, Xdata, Pdata, num_classes_);
  return true;
}


template<>
bool AddSpatialSoftmaxGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);  // Probabilities from softmax
  auto& dY = Input(1);
  auto* dX = Output(0);

  DCHECK_EQ(Y.ndim(), 4);

  const int N = Y.dim32(0);
  const int DP = Y.dim32(1);
  const int H = Y.dim32(2);
  const int W = Y.dim32(3);
  const int pixels = H * W;
  // there is a default one -- 0
  const int A = DP / num_classes_;
  const int D = DP - A;

  dX->Resize(N, D, H, W);

  const int size_sum_probs = N * A * pixels;
  if (sum_probs_.size() != size_sum_probs) {
    sum_probs_.Resize(size_sum_probs);
  }

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  float* sum_probs_data = sum_probs_.mutable_data<float>();

  // Complete math:
  // J_ij = h_i (delta_ij - h_j)
  // d x_i = sum_j d h_ij = sum_j J_ij * dy_j
  //       = sum_j h_i (delta_ij - h_j) * dy_j
  //       = h_i dy_i - (sum_j h_i h_j dy_j)
  //       = h_i dy_i - h_i sum_j h_j dy_j

  // Step 0: dx = dy
  DeCopyKernel<<<CAFFE_GET_BLOCKS(dX->size()), CAFFE_CUDA_NUM_THREADS, 0,
                   context_.cuda_stream()>>>(dX->size(), A, pixels, 
                                            dYdata, dXdata, num_classes_);

  // Step 1: s = Sum(dY[j] * Y[j])
  SumProbsKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
                   context_.cuda_stream()>>>(
    N, A, pixels, Ydata, dYdata, sum_probs_data, num_classes_);

  // Step 2: dX[i] = dX[i] - s
  SubSumKernel<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS, 0,
                   context_.cuda_stream()>>>(
    N, A, pixels, sum_probs_.data<float>(), dXdata, num_classes_-1);

  // Step 3: dX[i] = Y[i] * dX[i]
  DeMulKernel<<<CAFFE_GET_BLOCKS(dX->size()), CAFFE_CUDA_NUM_THREADS, 0,
                   context_.cuda_stream()>>>(dX->size(), A, pixels, 
                                            Ydata, dXdata, num_classes_);

  return true;
}


REGISTER_CUDA_OPERATOR(AddSpatialSoftmax,
                       AddSpatialSoftmaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(AddSpatialSoftmaxGradient,
                       AddSpatialSoftmaxGradientOp<float, CUDAContext>);
} // namespace caffe2