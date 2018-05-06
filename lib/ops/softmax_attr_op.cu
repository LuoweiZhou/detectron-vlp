#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "softmax_attr_op.h"

namespace caffe2 {

namespace {

__global__ void SoftmaxKernel(const int N, const int D, 
    const float* Xdata, float* Pdata) {
  CUDA_1D_KERNEL_LOOP(n, N) {
    const int ri = n * D;
    // get max value
    float max_val = -FLT_MAX;
    for (int i=0; i<D; i++) {
      max_val = max(max_val, Xdata[ri + i]);
    }
    // subtract and exponentiate
    float expsum = 0.0f;
    for (int i=0; i<D; i++) {
      const int idx = ri + i;
      const float expx = exp(Xdata[idx] - max_val);
      Pdata[idx] = expx;
      expsum += expx;
    }
    // normalize
    for (int i=0; i<D; i++) {
      Pdata[ri + i] /= expsum;
    }
  }
}

__global__ void SoftmaxAttrKernel(
    const int N, const int D, const int G, const int ignore,
    const float* Pdata, const int* Tdata, float* losses, float* weights) {
  CUDA_1D_KERNEL_LOOP(n, N) {
    const int ri = n * D;
    const int gi = n * G;

    // check all the labels
    int cnt = 0;
    for (int i=0; i<G; i++) {
      const int label = Tdata[gi + i];
      if (label != ignore) {
        cnt ++;
      } else {
        // we always fill in the first slots first
        break;
      }
    }
    float loss = 0.;
    float weight = 0.;
    if (cnt > 0) {
      for (int i=0; i<cnt; i++) {
        const int label = Tdata[gi + i];
        loss -= log(max(Pdata[ri + label], FLT_MIN));
      }
      loss /= static_cast<float>(cnt);
      // remember to add up the count
      weight = 1.;
    }

    losses[n] = loss;
    weights[n] = weight;
  }
}

__global__ void DivisionKernel(
    const int N, const float* sum_weight, float* losses) {
  CUDA_1D_KERNEL_LOOP(n, N) {
    if ((*sum_weight) > 0) {
      losses[n] /= *sum_weight;
    }
  }
}

__global__ void SoftmaxAttrGradientKernel(
    const int N, const int D, const int G, const int ignore,
    const float* Pdata, const int* Tdata, float* dXdata, float* wdata) {
  CUDA_1D_KERNEL_LOOP(n, N) {
    const int ri = n * D;
    const int gi = n * G;

    int cnt = 0;
    for (int i=0; i<G; i++) {
      const int label = Tdata[gi + i];
      if (label != ignore) {
        cnt ++;
      } else {
        // we always fill in the first slots first
        break;
      }
    }
    if (cnt > 0) {
      const float minus = 1. / static_cast<float>(cnt);
      for (int i=0; i<cnt; i++) {
        const int label = Tdata[gi + i];
        dXdata[ri + label] -= minus;
      }
      wdata[n] = 1.;
    } else {
      wdata[n] = 0.;
    }
  }
}

__global__ void DivisionBackKernel(
    const int N, const int D, const float* sum_weight, 
    const float* weights, const float* dlosses,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(index, N * D) {
    const int n = index / D;

    if (weights[n] > 0) {
      dXdata[index] *= (*dlosses) / (*sum_weight);
    } else {
      // no attribute label, should not go back
      dXdata[index] = 0.;
    }
  }
}

} // namespace


template <>
bool SoftmaxAttrOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);         // Logits
  auto& T = Input(1);         // Labels
  auto* P = Output(0);        // softmax probability, going to be re-used in gradient
  auto* avg_loss = Output(1); // average loss as output

  const int N = X.dim32(0);
  const int D = X.dim32(1);
  DCHECK_EQ(T.dim32(0), N);
  const int G = T.dim32(1);

  losses_.Resize(N);
  weights_.Resize(N);
  P->Resize(N, D);
  // should mean empty here
  avg_loss->Resize(vector<TIndex>());
  sum_weight_.Resize(1);

  const float* Xdata = X.data<float>();
  const int* Tdata = T.data<int>();
  float* Pdata = P->mutable_data<float>();
  float* ldata = losses_.mutable_data<float>();
  float* wdata = weights_.mutable_data<float>();

  //  First compute the softmax
  SoftmaxKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(N, D, Xdata, Pdata);

  // Compute loss
  SoftmaxAttrKernel
  <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
      0, context_.cuda_stream()>>>(
        N, D, G, ignore_, Pdata, Tdata, ldata, wdata);

  // summing up the weights
  float* sum_weight_data = sum_weight_.mutable_data<float>();
  math::Sum<float, CUDAContext>(N, wdata, sum_weight_data, &context_);

  // divide by weight if necessary, this is already taking the mean
  DivisionKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
      0, context_.cuda_stream()>>>(N, sum_weight_data, ldata);

  // sum the losses
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(N, ldata, avg_loss_data, &context_);

  math::Scale<float, CUDAContext>(
      1, scale_, avg_loss->data<float>(), avg_loss_data, &context_);

  return true;
}


template<>
bool SoftmaxAttrGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);    // Logits
  auto& T = Input(1);    // Label
  auto& P = Input(2);    // Softmax Probability
  auto& d_avg_loss = Input(3);
  auto* dX = Output(0);  // gradient wrt logits

  const int N = X.dim32(0);
  const int D = X.dim32(1);
  const int G = T.dim32(1);

  dX->ResizeLike(X);
  weights_.Resize(N);
  sum_weight_.Resize(1);

  const float* Xdata = X.data<float>();
  const int* Tdata = T.data<int>();
  const float* Pdata = P.data<float>();
  float* dXdata = dX->mutable_data<float>();
  float* wdata = weights_.mutable_data<float>();

  // copy the activations
  context_.Copy<float, CUDAContext, CUDAContext>(N*D, Pdata, dXdata);

  SoftmaxAttrGradientKernel
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, D, G, ignore_, Pdata, Tdata, dXdata, wdata);

  // summing up the weights
  float* sum_weight_data = sum_weight_.mutable_data<float>();
  math::Sum<float, CUDAContext>(N, wdata, sum_weight_data, &context_);

  const float* d_avg_loss_data = d_avg_loss.data<float>();
  DivisionBackKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
          N, D, sum_weight_data, wdata, d_avg_loss_data, dXdata);

  math::Scale<float, CUDAContext>(
    dX->size(), scale_, dX->data<float>(), dXdata, &context_);

  return true;
}


REGISTER_CUDA_OPERATOR(SoftmaxAttr,
                       SoftmaxAttrOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SoftmaxAttrGradient,
                       SoftmaxAttrGradientOp<float, CUDAContext>);
} // namespace caffe2