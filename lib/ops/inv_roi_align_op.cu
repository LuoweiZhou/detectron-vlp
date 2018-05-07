#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "inv_roi_align_op.h"

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
__device__ void bilinear_interpolate_forward(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__device__ T bilinear_interpolate_backward(const T* bottom_data,
    const int height, const int width,
    T y, T x,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void InvRoIAlignForward(const int nthreads, 
    const T* bottom_data, const float* bottom_rois, 
    const float spatial_scale, 
    const int batch, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    if (roi_batch_ind < 0 || roi_batch_ind >= batch) {
      continue;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1 (should not happen as I have removed those)
    T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(1.));
    T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(1.));
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_top_data = top_data + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_bottom_data = bottom_data + top_offset;
    const T bottom_data_this_bin = offset_bottom_data[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy ++) { // e.g., iy = 0, 1 
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++) {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_forward(height, width, y, x,
                                      w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high,
                                      index);

        T g1 = bottom_data_this_bin * w1 / count;
        T g2 = bottom_data_this_bin * w2 / count;
        T g3 = bottom_data_this_bin * w3 / count;
        T g4 = bottom_data_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          gpu_atomic_add(static_cast<T>(g1), offset_top_data + y_low * width + x_low);
          gpu_atomic_add(static_cast<T>(g2), offset_top_data + y_low * width + x_high);
          gpu_atomic_add(static_cast<T>(g3), offset_top_data + y_high * width + x_low);
          gpu_atomic_add(static_cast<T>(g4), offset_top_data + y_high * width + x_high);
        } // if
      } // ix
    } // iy
  }
}

template <typename T>
__global__ void InvRoIAlignBackward(const int nthreads, 
    const T* input_grad, const float* bottom_rois,
    const float spatial_scale, 
    const int batch, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    T* output_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    const int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    if (roi_batch_ind < 0 || roi_batch_ind >= batch) {
      output_grad[index] = 0.;
      continue;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(1.));
    T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(1.));
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input_grad = input_grad + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy ++) { // e.g., iy = 0, 1
      const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix ++) {
        const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate_backward(offset_input_grad, 
                                              height, 
                                              width, 
                                              y, 
                                              x, 
                                              index);
        output_val += val;
      }
    }
    output_val /= count;

    output_grad[index] = output_val;
  } // CUDA_1D_KERNEL_LOOP
} // InvRoIAlignBackward

} // namespace

template<>
bool InvRoIAlignOp<float, CUDAContext>::RunOnDevice() {
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

  InvRoIAlignForward<float>
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
bool InvRoIAlignGradientOp<float, CUDAContext>::RunOnDevice() {
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

  InvRoIAlignBackward<float>
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


REGISTER_CUDA_OPERATOR(InvRoIAlign,
                       InvRoIAlignOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(InvRoIAlignGradient,
                       InvRoIAlignGradientOp<float, CUDAContext>);
} // namespace caffe2