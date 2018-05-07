#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "resize_bilinear_as_op.h"

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
__global__ void ResizeBilinearAsForward(const int nthreads, const T* bottom_data,
    const int channels, const int in_height, const int in_width,
    const float inv_spatial_scale, const int height, const int width,
    T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idx = index;
    const int x = idx % width;
    idx /= width;
    const int y = idx % height;
    idx /= height;
    const int c = idx % channels;
    const int b = idx / channels;

    const float in_y = y * inv_spatial_scale;
    const int top_y_index = floorf(in_y);
    const int bottom_y_index =
        (in_y < in_height - 1) ? ceilf(in_y) : in_height - 1;
    const float y_lerp = in_y - top_y_index;

    const float in_x = x * inv_spatial_scale;
    const int left_x_index = floorf(in_x);
    const int right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - left_x_index;

    const int base_idx = (b * channels + c) * in_height * in_width;
    const int top_offset = top_y_index * in_width;
    const int bottom_offset = bottom_y_index * in_width;

    // also includes the case of extrapolation
    float top_left, top_right, bottom_left, bottom_right;
    if (top_y_index >= 0 && top_y_index < height) {
      if (left_x_index >= 0 && left_x_index < width) 
        top_left = bottom_data[base_idx + top_offset + left_x_index];
      else
        top_left = 0.;
      if (right_x_index >= 0 && right_x_index < width)
        top_right = bottom_data[base_idx + top_offset + right_x_index];
      else
        top_right = 0.;
    } else {
      top_left = 0.;
      top_right = 0.;
    }

    if (bottom_y_index >= 0 && bottom_y_index < height) {
      if (left_x_index >= 0 && left_x_index < width)
        bottom_left = bottom_data[base_idx + bottom_offset + left_x_index];
      else
        bottom_left = 0.;
      if (right_x_index >= 0 && right_x_index < width)
        bottom_right = bottom_data[base_idx + bottom_offset + right_x_index];
      else
        bottom_right = 0.;
    } else {
      bottom_left = 0.;
      bottom_right = 0.;
    }

    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    output[index] = top + (bottom - top) * y_lerp;
  }
}

template <typename T>
__global__ void ResizeBilinearAsBackward(const int nthreads, const T* input_grad,
    const int channels, const int in_height, const int in_width,
    const float inv_spatial_scale, const int output_height, const int output_width, 
    T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idx = index;
    const int x = idx % output_width;
    idx /= output_width;
    const int y = idx % output_height;
    idx /= output_height;
    const int c = idx % channels;
    const int b = idx / channels;

    const float original_y = y * inv_spatial_scale;
    const int top_y_index = floorf(original_y);
    const int bottom_y_index = (original_y < in_height - 1)
                                   ? ceilf(original_y)
                                   : in_height - 1;
    const float y_lerp = original_y - top_y_index;

    const float original_x = x * inv_spatial_scale;
    const int left_x_index = floorf(original_x);
    const int right_x_index = (original_x < in_width - 1)
                                  ? ceilf(original_x)
                                  : in_width - 1;
    const float x_lerp = original_x - left_x_index;

    const int base_idx = (b * channels + c) * in_height * in_width;
    const int top_offset = top_y_index * in_width;
    const int bottom_offset = bottom_y_index * in_width;

    if (top_y_index >= 0 && top_y_index < in_height) {
      const float dtop = (1 - y_lerp) * input_grad[index];
      if (left_x_index >= 0 && left_x_index < in_width)
        gpu_atomic_add(static_cast<T>((1 - x_lerp) * dtop), 
                      output_grad + (base_idx + top_offset + left_x_index));
      if (right_x_index >= 0 && right_x_index < in_width)
        gpu_atomic_add(static_cast<T>(x_lerp * dtop),
                      output_grad + (base_idx + top_offset + right_x_index));
    }

    if (bottom_y_index >= 0 && bottom_y_index < in_height) {
      const float dbottom = y_lerp * input_grad[index];
      if (left_x_index >= 0 && left_x_index < in_width) 
        gpu_atomic_add(static_cast<T>((1 - x_lerp) * dbottom), 
                      output_grad + (base_idx + bottom_offset + left_x_index));
      if (right_x_index >= 0 && right_x_index < in_width)
        gpu_atomic_add(static_cast<T>(x_lerp * dbottom), 
                      output_grad + (base_idx + bottom_offset + right_x_index));
    }
  }
}

} // namespace

template<>
bool ResizeBilinearAsOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data
  auto& R = Input(1);  // Reference region
  auto* Y = Output(0); // Output data

  const float inv_spatial_scale = 1. / spatial_scale_;

  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);

  const int pH = R.dim32(2);
  const int pW = R.dim32(3);

  Y->ResizeLike(R);

  int output_size = Y->size();
  ResizeBilinearAsForward<float><<<CAFFE_GET_BLOCKS(output_size),
                          CAFFE_CUDA_NUM_THREADS,
                        0, context_.cuda_stream()>>>(
                          output_size, X.data<float>(), C, H, W, 
                          inv_spatial_scale, pH, pW, Y->mutable_data<float>());

  return true;
}


template<>
bool ResizeBilinearAsGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY  = Input(0);  // Gradient of the output data
  auto& X  = Input(1);  // Input data
  auto& R = Input(2);
  auto* dX = Output(0); // Gradient of the input data

  const float inv_spatial_scale = 1. / spatial_scale_;

  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);

  const int pH = R.dim32(2);
  const int pW = R.dim32(3);

  dX->ResizeLike(X);
  // Must zero-out dX before accumulating gradients
  math::Set<float, CUDAContext>(dX->size(), 0.f, dX->mutable_data<float>(), &context_);

  ResizeBilinearAsBackward<float><<<CAFFE_GET_BLOCKS(dY.size()),
                           CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
                            dY.size(), dY.data<float>(),
                            C, H, W, 
                            inv_spatial_scale, pH, pW,
                            dX->mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(ResizeBilinearAs,
                       ResizeBilinearAsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ResizeBilinearAsGradient,
                       ResizeBilinearAsGradientOp<float, CUDAContext>);
} // namespace caffe2