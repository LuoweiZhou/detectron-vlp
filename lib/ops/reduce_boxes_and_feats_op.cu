#include <cfloat>
#include <thrust/sort.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "reduce_boxes_and_feats_op.h"

namespace caffe2 {

namespace {

__device__ void _copy(const int size, const float* source, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = *(source++);
    }
}

__global__ void GetValuesAndIndices(const int nthreads,
                                      const float* boxes,
                                      float* values,
                                      TIndex* indices) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    values[index] = boxes[index * 6 + 4];
    indices[index] = static_cast<TIndex>(index);
  }
}

__global__ void CopyBoxes(const int nthreads,
                          const float* boxes,
                          const float im,
                          float* output_boxes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int bid = index * 6;
    const int oid = index * 5;
    output_boxes[oid] = im;
    _copy(4, boxes + bid, output_boxes + oid + 1);
  }
}

__global__ void GetBoxes(const int nthreads,
                         const float* boxes,
                         const TIndex* indices,
                         const float im,
                         float* output_boxes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int bid = static_cast<int>(indices[index]) * 6;
    const int oid = index * 5;
    output_boxes[oid] = im;
    _copy(4, boxes + bid, output_boxes + oid + 1);
  }
}

__global__ void GetFeats(const int nthreads,
                         const float* feats,
                         const TIndex* indices,
                         const int num_feat,
                         float* output_feats) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_feat;
    const int i = index / num_feat;
    const int ii = static_cast<int>(indices[i]);
    const int oid = ii * num_feat + j;
    output_feats[index] = feats[oid];
  }
}

} // namespace

template<>
bool ReduceBoxesAndFeatsOp<float, CUDAContext>::RunOnDevice() {
  auto& boxes = Input(0);
  const int num_total = boxes.dim32(0);
  const float* boxes_pointer = boxes.data<float>();
  auto& feats = Input(1);
  DCHECK_EQ(feats.dim32(0), num_total);
  const int num_feat = feats.dim32(1);
  const float* feats_pointer = feats.data<float>();

  if (num_total == 0) {
    Output(0)->Resize(0, 5);
    Output(1)->Resize(0, num_feat);
    Output(0)->mutable_data<float>();
    Output(1)->mutable_data<float>();
    return true;
  } else if (num_total <= dpi_) {
    // just return all the detections
    auto* output_boxes = Output(0);
    output_boxes->Resize(num_total, 5);
    CopyBoxes<<<CAFFE_GET_BLOCKS(num_total), CAFFE_CUDA_NUM_THREADS,
                  0, context_.cuda_stream()>>>(num_total, boxes_pointer, im_float_,
                                              output_boxes->mutable_data<float>());
    auto* output_feats = Output(1);
    output_feats->ResizeLike(feats);
    context_.Copy<float, CUDAContext, CUDAContext>(num_total * num_feat, feats_pointer, 
                                                  output_feats->mutable_data<float>());
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
  output_boxes->Resize(dpi_, 5);
  auto* output_feats = Output(1);
  output_feats->Resize(dpi_, num_feat);

  GetBoxes<<<CAFFE_GET_BLOCKS(dpi_), CAFFE_CUDA_NUM_THREADS,
            0, context_.cuda_stream()>>>(dpi_, boxes_pointer, indices_pointer,
                                        im_float_, output_boxes->mutable_data<float>());

  const int dpi_feat = dpi_ * num_feat;
  GetFeats<<<CAFFE_GET_BLOCKS(dpi_feat), CAFFE_CUDA_NUM_THREADS,
            0, context_.cuda_stream()>>>(dpi_feat, feats_pointer, indices_pointer,
                                         num_feat, output_feats->mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(ReduceBoxesAndFeats,
                       ReduceBoxesAndFeatsOp<float, CUDAContext>);
} // namespace caffe2