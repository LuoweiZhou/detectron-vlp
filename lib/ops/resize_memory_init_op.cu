#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "resize_memory_init_op.h"

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
__global__ void ResizeMemoryInitForward(const int nthreads, const T* images,
    const int channels, const int in_height, const int in_width,
    const int output_height, const int output_width, const float spatial_scale,
    const float e_value, const float* image_info, T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idx = index;
    const int x = idx % output_width;
    idx /= output_width;
    const int y = idx % output_height;
    idx /= output_height;
    const int c = idx % channels;
    const int b = idx / channels;

    const float im_height = image_info[b * 3];
    const float im_width = image_info[b * 3 + 1];

    const float in_y = y / (im_height * spatial_scale) * in_height;
    const float in_x = x / (im_width * spatial_scale) * in_width;

    // only do interpolation, no extrapolation
    if (in_y < 0 || in_y > in_height - 1 || in_x < 0 || in_x > in_width - 1) {
      output[index] = e_value;
      continue;
    }

    const int top_y_index = floorf(in_y);
    const int bottom_y_index = ceilf(in_y);
    const float y_lerp = in_y - top_y_index;

    const int left_x_index = floorf(in_x);
    const int right_x_index = ceilf(in_x);
    const float x_lerp = in_x - left_x_index;

    const int base_idx = c * in_height * in_width;
    const int top_offset = top_y_index * in_width;
    const int bottom_offset = bottom_y_index * in_width;

    const float top_left(images[base_idx + top_offset + left_x_index]);
    const float top_right(images[base_idx + top_offset + right_x_index]);
    const float bottom_left(images[base_idx + bottom_offset + left_x_index]);
    const float bottom_right(images[base_idx + bottom_offset + right_x_index]);

    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    output[index] = top + (bottom - top) * y_lerp;
  }
}

template <typename T>
__global__ void ResizeMemoryInitBackward(const int nthreads, const T* input_grad,
    const int channels, const int in_height, const int in_width,
    const int output_height, const int output_width, const float spatial_scale,
    const float* image_info, T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idx = index;
    const int x = idx % output_width;
    idx /= output_width;
    const int y = idx % output_height;
    idx /= output_height;
    const int c = idx % channels;
    const int b = idx / channels;

    const float im_height = image_info[b * 3];
    const float im_width = image_info[b * 3 + 1];

    const float original_y = y / (im_height * spatial_scale) * in_height;
    const float original_x = x / (im_width * spatial_scale) * in_width;

    // only do interpolation, no extrapolation
    if (original_y >= 0 && original_y <= in_height - 1 && original_x >= 0 && original_x <= in_width - 1) {
      const int top_y_index = floorf(original_y);
      const int bottom_y_index = ceilf(original_y);
      const float y_lerp = original_y - top_y_index;

      const int left_x_index = floorf(original_x);
      const int right_x_index = ceilf(original_x);
      const float x_lerp = original_x - left_x_index;

      const int base_idx = c * in_height * in_width;
      const int top_offset = top_y_index * in_width;
      const int bottom_offset = bottom_y_index * in_width;

      const float dtop = (1 - y_lerp) * input_grad[index];
      gpu_atomic_add(static_cast<T>((1 - x_lerp) * dtop), 
                    output_grad + (base_idx + top_offset + left_x_index));
      gpu_atomic_add(static_cast<T>(x_lerp * dtop),
                    output_grad + (base_idx + top_offset + right_x_index));

      const float dbottom = y_lerp * input_grad[index];
      gpu_atomic_add(static_cast<T>((1 - x_lerp) * dbottom), 
                    output_grad + (base_idx + bottom_offset + left_x_index));
      gpu_atomic_add(static_cast<T>(x_lerp * dbottom), 
                    output_grad + (base_idx + bottom_offset + right_x_index));
    }
  }
}

} // namespace

template<>
bool ResizeMemoryInitOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data
  auto& im_info = Input(1); // Image information
  auto& data = Input(2); // Reference blob
  auto* Y = Output(0); // Output data

  const float spatial_scale = spatial_scale_; // Spatial scale
  // N, C, H, W from C, H', W'
  const int number = data.dim32(0);
  const int height = data.dim32(2);
  const int width = data.dim32(3);
  // channel is changed
  Y->Resize(number, X.dim32(0), height, width);
  int output_size = Y->size();

  ResizeMemoryInitForward<float><<<CAFFE_GET_BLOCKS(output_size),
                          CAFFE_CUDA_NUM_THREADS,
                          0, context_.cuda_stream()>>>(
                          output_size, X.data<float>(), 
                          X.dim32(0), X.dim32(1), X.dim32(2), 
                          height, width, spatial_scale,
                          e_value_, im_info.data<float>(), 
                          Y->mutable_data<float>());

  return true;
}


template<>
bool ResizeMemoryInitGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY  = Input(0); // Gradient of the output data
  auto& X  = Input(1); // Input data
  auto& im_info = Input(2); // Image information
  auto& data = Input(3); // Reference blob
  auto* dX = Output(0); // Gradient of the input data
  
  const float spatial_scale = spatial_scale_; // Spatial scale
  // N, C, H, W
  const int height = data.dim32(2);
  const int width = data.dim32(3);

  dX->ResizeLike(X);
  // Must zero-out dX before accumulating gradients
  math::Set<float, CUDAContext>(dX->size(), 0.f, dX->mutable_data<float>(), &context_);

  ResizeMemoryInitBackward<float><<<CAFFE_GET_BLOCKS(dY.size()),
                           CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
                          dY.size(), dY.data<float>(),
                          X.dim32(0), X.dim32(1), X.dim32(2), 
                          height, width, spatial_scale,
                          im_info.data<float>(),
                          dX->mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(ResizeMemoryInit,
                       ResizeMemoryInitOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ResizeMemoryInitGradient,
                       ResizeMemoryInitGradientOp<float, CUDAContext>);
} // namespace caffe2