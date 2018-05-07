#include <cfloat>
#include <thrust/sort.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "reduce_boxes_only_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__
int gpu_atomic_add(const int val, int* address) {
  return atomicAdd(address, val);
}

__device__ void _copy(const int size, const float* source, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

__device__ float _assign_level(const float area, const int k_min, const int k_max, const float s0, const float k) {
  const float s = sqrt(area);
  float lvl = floor(log2(s / s0 + 1e-6)) + k;
  lvl = (lvl < k_min) ? k_min : lvl;
  lvl = (lvl > k_max) ? k_max : lvl;
  return lvl;
}

__global__ void GetValuesAndIndices(const int nthreads,
                                      const float* boxes,
                                      float* values,
                                      TIndex* indices) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    values[index] = boxes[index * 12 + 4];
    indices[index] = static_cast<TIndex>(index);
  }
}

__global__ void CopyBoxes(const int nthreads,
                          const float* boxes,
                          const float im,
                          const int k_min,
                          const int k_max,
                          const float c_scale,
                          const float c_level,
                          float* output_boxes,
                          int* output_stats) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int bid = index * 12;
    const int oid = index * 11;
    output_boxes[oid] = im;
    _copy(4, boxes + bid, output_boxes + oid + 1);
    // do the calculation of assigning levels
    const float lvl_assign = _assign_level(boxes[bid + 5], k_min, k_max, 
                                          c_scale, c_level);
    output_boxes[oid + 5] = lvl_assign;
    const int lvl = static_cast<int>(lvl_assign) - k_min;
    gpu_atomic_add(1, output_stats + lvl);
    output_boxes[oid + 6] = boxes[bid + 6];
    _copy(4, boxes + bid + 8, output_boxes + oid + 7);
  }
}

__global__ void GetBoxes(const int nthreads,
                         const float* boxes,
                         const TIndex* indices,
                         const float im,
                         const int k_min,
                         const int k_max,
                         const float c_scale,
                         const float c_level,
                         float* output_boxes,
                         int* output_stats) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int bid = static_cast<int>(indices[index]) * 12;
    const int oid = index * 11;
    output_boxes[oid] = im;
    _copy(4, boxes + bid, output_boxes + oid + 1);
    // do the calculation of assigning levels
    const float lvl_assign = _assign_level(boxes[bid + 5], k_min, k_max, 
                                          c_scale, c_level);
    output_boxes[oid + 5] = lvl_assign;
    const int lvl = static_cast<int>(lvl_assign) - k_min;
    gpu_atomic_add(1, output_stats + lvl);
    output_boxes[oid + 6] = boxes[bid + 6];
    _copy(4, boxes + bid + 8, output_boxes + oid + 7);
  }
}

} // namespace

template<>
bool ReduceBoxesOnlyOp<float, CUDAContext>::RunOnDevice() {
  auto& boxes = Input(0);
  const int num_total = boxes.dim32(0);
  const int num_levels = k_max_ - k_min_ + 1;
  const float* boxes_pointer = boxes.data<float>();

  if (num_total == 0) {
    Output(0)->Resize(0, 11);
    Output(1)->Resize(num_levels);
    Output(0)->mutable_data<float>();
    int* output_stats_pointer = Output(1)->mutable_data<int>();
    math::Set<int, CUDAContext>(num_levels, 0, output_stats_pointer, &context_);
    return true;
  } else if (num_total <= dpi_) {
    // just return all the detections
    auto* output_boxes = Output(0);
    output_boxes->Resize(num_total, 11);
    auto* output_stats = Output(1);
    output_stats->Resize(num_levels);
    math::Set<int, CUDAContext>(num_levels, 0, output_stats->mutable_data<int>(), &context_);

    CopyBoxes<<<CAFFE_GET_BLOCKS(num_total), CAFFE_CUDA_NUM_THREADS,
                  0, context_.cuda_stream()>>>(num_total, boxes_pointer, im_float_,
                                              k_min_, k_max_, c_scale_f_, c_level_f_,
                                              output_boxes->mutable_data<float>(),
                                              output_stats->mutable_data<int>());
    return true;
  }

  values.Resize(num_total);
  indices.Resize(num_total);
  float* values_pointer = values.mutable_data<float>();
  TIndex* indices_pointer = indices.mutable_data<TIndex>();

  GetValuesAndIndices<<<CAFFE_GET_BLOCKS(num_total), CAFFE_CUDA_NUM_THREADS,
                                0, context_.cuda_stream()>>>(num_total,
                                                            boxes_pointer,
                                                            values_pointer,
                                                            indices_pointer);

  thrust::sort_by_key(thrust::cuda::par.on(context_.cuda_stream()),
                                values_pointer,
                                values_pointer + num_total,
                                indices_pointer,
                                thrust::greater<float>());

  auto* output_boxes = Output(0);
  output_boxes->Resize(dpi_, 11);
  auto* output_stats = Output(1);
  output_stats->Resize(num_levels);
  math::Set<int, CUDAContext>(num_levels, 0, output_stats->mutable_data<int>(), &context_);

  GetBoxes<<<CAFFE_GET_BLOCKS(dpi_), CAFFE_CUDA_NUM_THREADS,
            0, context_.cuda_stream()>>>(dpi_, boxes_pointer, indices_pointer,
                                        im_float_, k_min_, k_max_,
                                        c_scale_f_, c_level_f_,
                                        output_boxes->mutable_data<float>(),
                                        output_stats->mutable_data<int>());

  return true;
}

REGISTER_CUDA_OPERATOR(ReduceBoxesOnly,
                       ReduceBoxesOnlyOp<float, CUDAContext>);
} // namespace caffe2