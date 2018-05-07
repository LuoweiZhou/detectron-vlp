#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "resize_bilinear_op.h"

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
__global__ void ResizeBilinearForward(const int nthreads, const T* images,
    const int channels, const int in_height, const int in_width,
    const float height_scale, const float width_scale,
    const int output_height, const int output_width,
    T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idx = index;
    const int x = idx % output_width;
    idx /= output_width;
    const int y = idx % output_height;
    idx /= output_height;
    const int c = idx % channels;
    const int b = idx / channels;

    const float in_y = y * height_scale;
    const int top_y_index = floorf(in_y);
    const int bottom_y_index =
        (in_y < in_height - 1) ? ceilf(in_y) : in_height - 1;
    const float y_lerp = in_y - top_y_index;

    const float in_x = x * width_scale;
    const int left_x_index = floorf(in_x);
    const int right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - left_x_index;

    const int base_idx = (b * channels + c) * in_height * in_width;
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
__global__ void ResizeBilinearBackward(const int nthreads, const T* input_grad,
    const int channels, const int in_height, const int in_width,
    const float height_scale, const float width_scale,
    const int output_height, const int output_width, T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idx = index;
    const int x = idx % output_width;
    idx /= output_width;
    const int y = idx % output_height;
    idx /= output_height;
    const int c = idx % channels;
    const int b = idx / channels;

    const float original_y = y * height_scale;
    const int top_y_index = floorf(original_y);
    const int bottom_y_index = (original_y < in_height - 1)
                                   ? ceilf(original_y)
                                   : in_height - 1;
    const float y_lerp = original_y - top_y_index;

    const float original_x = x * width_scale;
    const int left_x_index = floorf(original_x);
    const int right_x_index = (original_x < in_width - 1)
                                  ? ceilf(original_x)
                                  : in_width - 1;
    const float x_lerp = original_x - left_x_index;

    const int base_idx = (b * channels + c) * in_height * in_width;
    const int top_offset = top_y_index * in_width;
    const int bottom_offset = bottom_y_index * in_width;

    const float dtop = (1 - y_lerp) * input_grad[index];
    gpu_atomic_add(static_cast<T>((1 - x_lerp) * dtop), 
                    output_grad +
                    (base_idx + top_offset + left_x_index));
    gpu_atomic_add(static_cast<T>(x_lerp * dtop),
                    output_grad +
                    (base_idx + top_offset + right_x_index));

    const float dbottom = y_lerp * input_grad[index];
    gpu_atomic_add(static_cast<T>((1 - x_lerp) * dbottom), 
                    output_grad +
                    (base_idx + bottom_offset + left_x_index));
    gpu_atomic_add(static_cast<T>(x_lerp * dbottom), 
                    output_grad +
                    (base_idx + bottom_offset + right_x_index));
  }
}

} // namespace

template<>
bool ResizeBilinearOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data
  auto* Y = Output(0); // Output data

  const float height_scale = static_cast<float>(X.dim32(2)) / 
                             static_cast<float>(height_);
  const float width_scale = static_cast<float>(X.dim32(3)) / 
                            static_cast<float>(width_);

  // N, C, H, W
  Y->Resize(X.dim32(0), X.dim32(1), height_, width_);
  int output_size = Y->size();
  ResizeBilinearForward<float><<<CAFFE_GET_BLOCKS(output_size),
                          CAFFE_CUDA_NUM_THREADS,
                          0, context_.cuda_stream()>>>(
      output_size, X.data<float>(), X.dim32(1), X.dim32(2), X.dim32(3), 
      height_scale, width_scale, height_, width_, Y->mutable_data<float>());

  return true;
}


template<>
bool ResizeBilinearGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY  = Input(0);  // Gradient of the output data
  auto& X  = Input(1);  // Input data
  auto* dX = Output(0); // Gradient of the input data

  const float height_scale = static_cast<float>(X.dim32(2)) / 
                             static_cast<float>(height_);
  const float width_scale = static_cast<float>(X.dim32(3)) / 
                            static_cast<float>(width_);

  dX->ResizeLike(X);
  // Must zero-out dX before accumulating gradients
  math::Set<float, CUDAContext>(dX->size(), 0.f, dX->mutable_data<float>(), &context_);

  ResizeBilinearBackward<float><<<CAFFE_GET_BLOCKS(dY.size()),
                           CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
                          dY.size(), dY.data<float>(),
                          X.dim32(1), X.dim32(2), X.dim32(3), 
                          height_scale, width_scale, height_, width_,
                          dX->mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(ResizeBilinear,
                       ResizeBilinearOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ResizeBilinearGradient,
                       ResizeBilinearGradientOp<float, CUDAContext>);
} // namespace caffe2