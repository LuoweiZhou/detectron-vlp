#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "inv_crop_and_resize_op.h"

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
__global__ void InvCropAndResizeForward(const int nthreads, 
    const T* bottom_data, const float* bottom_rois, 
    const float spatial_scale, 
    const int batch, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int x = index % pooled_width;
    int y = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    if (roi_batch_ind < 0 || roi_batch_ind >= batch) {
      continue;
    }

    const float x1 = offset_bottom_rois[1];
    const float y1 = offset_bottom_rois[2];
    const float x2 = offset_bottom_rois[3];
    const float y2 = offset_bottom_rois[4];

    const float original_y = (y1 + (y2 - y1) * y / (pooled_height - 1)) * spatial_scale;
    const float original_x = (x1 + (x2 - x1) * x / (pooled_width - 1)) * spatial_scale;

    const int top_y_index = floorf(original_y);
    const int bottom_y_index = ceilf(original_y);
    const float y_lerp = original_y - top_y_index;

    const int left_x_index = floorf(original_x);
    const int right_x_index = ceilf(original_x);
    const float x_lerp = original_x - left_x_index;

    const int base_idx = c * height * width;
    const int top_offset = top_y_index * width;
    const int bottom_offset = bottom_y_index * width;

    if (top_y_index >= 0 && top_y_index < height) {
      const float dtop = (1 - y_lerp) * bottom_data[index];
      if (left_x_index >= 0 && left_x_index < width)
        gpu_atomic_add(static_cast<T>((1 - x_lerp) * dtop), 
                      top_data + (base_idx + top_offset + left_x_index));
      if (right_x_index >= 0 && right_x_index < width)
        gpu_atomic_add(static_cast<T>(x_lerp * dtop),
                      top_data + (base_idx + top_offset + right_x_index));
    }

    if (bottom_y_index >= 0 && bottom_y_index < height) {
      const float dbottom = y_lerp * bottom_data[index];
      if (left_x_index >= 0 && left_x_index < width) 
        gpu_atomic_add(static_cast<T>((1 - x_lerp) * dbottom), 
                      top_data + (base_idx + bottom_offset + left_x_index));
      if (right_x_index >= 0 && right_x_index < width)
        gpu_atomic_add(static_cast<T>(x_lerp * dbottom), 
                      top_data + (base_idx + bottom_offset + right_x_index));
    }
  }
}

template <typename T>
__global__ void InvCropAndResizeBackward(const int nthreads, 
    const T* input_grad, const float* bottom_rois,
    const float spatial_scale, 
    const int batch, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    const int x = index % pooled_width;
    const int y = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    const int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    if (roi_batch_ind < 0 || roi_batch_ind >= batch) {
      output_grad[index] = 0.;
      continue;
    }

    const float x1 = offset_bottom_rois[1];
    const float y1 = offset_bottom_rois[2];
    const float x2 = offset_bottom_rois[3];
    const float y2 = offset_bottom_rois[4];

    // the distance between two sampled points are n-1, rather than n
    const float in_y = (y1 + (y2 - y1) * y / (pooled_height - 1)) * spatial_scale;
    const float in_x = (x1 + (x2 - x1) * x / (pooled_width - 1)) * spatial_scale;

    const int top_y_index = floorf(in_y);
    const int bottom_y_index = ceilf(in_y);
    const float y_lerp = in_y - top_y_index;

    const int left_x_index = floorf(in_x);
    const int right_x_index = ceilf(in_x);
    const float x_lerp = in_x - left_x_index;

    const int base_idx = c * height * width;
    const int top_offset = top_y_index * width;
    const int bottom_offset = bottom_y_index * width;

    // also includes the case of extrapolation
    float top_left, top_right, bottom_left, bottom_right;
    if (top_y_index >= 0 && top_y_index < height) {
      if (left_x_index >= 0 && left_x_index < width) 
        top_left = input_grad[base_idx + top_offset + left_x_index];
      else
        top_left = 0.;
      if (right_x_index >= 0 && right_x_index < width)
        top_right = input_grad[base_idx + top_offset + right_x_index];
      else
        top_right = 0.;
    } else {
      top_left = 0.;
      top_right = 0.;
    }

    if (bottom_y_index >= 0 && bottom_y_index < height) {
      if (left_x_index >= 0 && left_x_index < width)
        bottom_left = input_grad[base_idx + bottom_offset + left_x_index];
      else
        bottom_left = 0.;
      if (right_x_index >= 0 && right_x_index < width)
        bottom_right = input_grad[base_idx + bottom_offset + right_x_index];
      else
        bottom_right = 0.;
    } else {
      bottom_left = 0.;
      bottom_right = 0.;
    }

    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

    output_grad[index] = top + (bottom - top) * y_lerp;
  } // CUDA_1D_KERNEL_LOOP
} // InvCropAndResizeBackward

} // namespace

template<>
bool InvCropAndResizeOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data
  auto& R = Input(1);  // RoIs
  auto& RX = Input(2); // RoI features
  auto* Y = Output(0); // RoI pooled data

  Y->ResizeLike(X);
  math::Set<float, CUDAContext>(
       Y->size(), 0.f, Y->mutable_data<float>(), &context_);

  // if R is empty, then just return zero
  if (R.size() == 0)
    return true;

  // get dimensions
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);

  DCHECK_EQ(C, RX.dim32(1));
  const int pH = RX.dim32(2);
  const int pW = RX.dim32(3);

  InvCropAndResizeForward<float>
      <<<CAFFE_GET_BLOCKS(RX.size()),
         CAFFE_CUDA_NUM_THREADS, 0,
      context_.cuda_stream()>>>(
          RX.size(),
          RX.data<float>(),
          R.data<float>(),
          spatial_scale_,
          N, C,
          H, W,
          pH, pW,
          Y->mutable_data<float>());
  return true;
}

template<>
bool InvCropAndResizeGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X  = Input(0);  // Input data
  auto& R  = Input(1);  // RoIs
  auto& RX = Input(2);  // RoI features
  auto& dY = Input(3);  // Gradient of net w.r.t. output of "forward" op
                        // (aka "gradOutput")
  auto* dRX = Output(0);// Gradient of net w.r.t. input to "forward" op
                        // (aka "gradInput")

  // get dimensions
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);

  const int pH = RX.dim32(2);
  const int pW = RX.dim32(3);

  dRX->ResizeLike(RX);

  if (R.size() == 0) {
    // The following mutable_data calls are needed to allocate the tensors
    dRX->mutable_data<float>();
    return true;
  }

  InvCropAndResizeBackward<float>
      <<<CAFFE_GET_BLOCKS(RX.size()),
         CAFFE_CUDA_NUM_THREADS, 0,
      context_.cuda_stream()>>>(
          RX.size(),
          dY.data<float>(),
          R.data<float>(),
          spatial_scale_,
          N, C,
          H, W,
          pH, pW,
          dRX->mutable_data<float>());

  return true;
}


REGISTER_CUDA_OPERATOR(InvCropAndResize,
                       InvCropAndResizeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(InvCropAndResizeGradient,
                       InvCropAndResizeGradientOp<float, CUDAContext>);
} // namespace caffe2