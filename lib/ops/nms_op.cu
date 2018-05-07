#include <cfloat>
#include <thrust/sort.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

#include "nms_op.h"

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
    values[index] = boxes[index * 8 + 4];
    indices[index] = static_cast<TIndex>(index);
  }
}

__global__ void ComputeOverlapping(const int nthreads,
                                      const float* input_boxes,
                                      const TIndex* indices,
                                      const int num_total,
                                      float* overlaps) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idA = index / num_total;
    int idB = index % num_total;

    // if idA < idB, then the score should be higher
    if (idA < idB) {
      const int iidA = static_cast<int>(indices[idA]) * 8;
      const int iidB = static_cast<int>(indices[idB]) * 8;

      const float x1A = input_boxes[iidA];
      const float y1A = input_boxes[iidA+1];
      const float x2A = input_boxes[iidA+2];
      const float y2A = input_boxes[iidA+3];
      const float areaA = input_boxes[iidA+5];

      const float x1B = input_boxes[iidB];
      const float y1B = input_boxes[iidB+1];
      const float x2B = input_boxes[iidB+2];
      const float y2B = input_boxes[iidB+3];
      const float areaB = input_boxes[iidB+5];

      const float xx1 = (x1A > x1B) ? x1A : x1B;
      const float yy1 = (y1A > y1B) ? y1A : y1B;
      const float xx2 = (x2A < x2B) ? x2A : x2B;
      const float yy2 = (y2A < y2B) ? y2A : y2B;

      float w = xx2 - xx1 + 1.;
      w = (w > 0.) ? w : 0.;
      float h = yy2 - yy1 + 1.;
      h = (h > 0.) ? h : 0.;
      const float inter = w * h;

      overlaps[idA * num_total + idB] = inter / (areaA + areaB - inter);
    } 
  }
}

__global__ void NMSForward(const int nthreads,
                          const float* overlaps,
                          const TIndex* indices,
                          const int num_total,
                          const float threshold,
                          const int top_n,
                          float* output_boxes,
                          int* output_index,
                          int* cnt) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    *cnt = 0;
    for (int i=0; i<num_total; i++) {
      const int id = static_cast<int>(indices[i]);
      // make sure we will change for every box
      if (output_boxes[id * 8 + 7] < 1.) {
        for (int j=i+1; j<num_total; j++) {
          if (overlaps[i * num_total + j] >= threshold) {
            const int jd = static_cast<int>(indices[j]);
            output_boxes[jd * 8 + 7] = 1.;
          }
        }
        // should be the actual index
        output_index[(*cnt)] = id;
        (*cnt)++;
      }
      // enough boxes, still assign box
      if ((*cnt) == top_n) {
        for (int j=i+1; j<num_total; j++) {
          const int jd = static_cast<int>(indices[j]);
          output_boxes[jd * 8 + 7] = 1.;
        }
        break;
      }
    }
  }
}

__global__ void CopyBoxes(const int nthreads,
                          const float* boxes,
                          float* output_boxes) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int bid = i * 8;
    const int oid = i * 6;
    _copy(5, boxes + bid, output_boxes + oid);
    output_boxes[oid+5] = boxes[bid+6];
  }
}

__global__ void NMSReduceBoxes(const int nthreads,
                          const float* boxes,
                          const int* index,
                          float* output_boxes) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int id = index[i];
    const int bid = id * 8;
    const int oid = i * 6;
    _copy(5, boxes + bid, output_boxes + oid);
    output_boxes[oid+5] = boxes[bid+6];
  }
}

__global__ void NMSReduceFeats(const int nthreads,
                          const float* feats,
                          const int* index,
                          const int num_feat,
                          float* output_feats) {
  CUDA_1D_KERNEL_LOOP(ii, nthreads) {
    const int j = ii % num_feat;
    const int i = ii / num_feat;
    const int id = index[i];

    output_feats[ii] = feats[id * num_feat + j];
  }
}

} // namespace

template<>
bool NMSOp<float, CUDAContext>::RunOnDevice() {
  auto& boxes = Input(0);
  DCHECK_EQ(boxes.dim32(1), 8);
  const int num_total = boxes.dim32(0);
  auto& feats = Input(1);
  DCHECK_EQ(feats.dim32(0), num_total);
  const int num_feat = feats.dim32(1);
  // handle the empty case
  if (num_total == 0) {
    Output(0)->Resize(0, 6);
    Output(0)->mutable_data<float>();
    Output(1)->Resize(0, num_feat);
    Output(1)->mutable_data<float>();
    return true;
  } else if (num_total == 1) {
    auto* output_boxes = Output(0);
    auto* output_feats = Output(1);
    output_boxes->Resize(1, 6);
    output_feats->Resize(1, num_feat);
    CopyBoxes<<<CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS,
                  0, context_.cuda_stream()>>>(1,
                                               boxes.data<float>(),
                                               output_boxes->mutable_data<float>());
    context_.Copy<float, CUDAContext, CUDAContext>(num_feat, feats.data<float>(), 
                                                  output_feats->mutable_data<float>());
    return true;
  }

  const int num_pair = num_total * num_total;
  const float* boxes_pointer = boxes.data<float>();
  const float* feats_pointer = feats.data<float>();

  values.Resize(num_total);
  indices.Resize(num_total);
  float* values_pointer = values.mutable_data<float>();
  TIndex* indices_pointer = indices.mutable_data<TIndex>();

  GetValuesAndIndices<<<CAFFE_GET_BLOCKS(num_total), CAFFE_CUDA_NUM_THREADS,
                                0, context_.cuda_stream()>>>(num_total,
                                                            boxes_pointer,
                                                            values_pointer,
                                                            indices_pointer);

  // sort the value and get the indexes
  thrust::sort_by_key(thrust::cuda::par.on(context_.cuda_stream()),
                                values_pointer,
                                values_pointer + num_total,
                                indices_pointer,
                                thrust::greater<float>());

  // pairwise comparison
  overlaps.Resize(num_total, num_total);
  float* overlaps_pointer = overlaps.mutable_data<float>();
  // initialize everything
  math::Set<float, CUDAContext>(num_pair, 0., overlaps_pointer, &context_);

  ComputeOverlapping<<<CAFFE_GET_BLOCKS(num_pair), CAFFE_CUDA_NUM_THREADS,
                                0, context_.cuda_stream()>>>(num_pair,
                                                            boxes_pointer,
                                                            indices_pointer,
                                                            num_total,
                                                            overlaps_pointer);
  // then just reduce by setting up the index
  middle.ResizeLike(boxes);
  float* middle_pointer = middle.mutable_data<float>();
  context_.Copy<float, CUDAContext, CUDAContext>(num_total * 8, boxes_pointer, 
                                                middle_pointer);

  // also remember the index
  mindex.Resize(num_total);
  int* mindex_pointer = mindex.mutable_data<int>();
  math::Set<int, CUDAContext>(num_total, -1, mindex_pointer, &context_);

  mcounter.Resize(1);
  int* mcounter_pointer = mcounter.mutable_data<int>();

  // using one thread to go down the list
  NMSForward<<<CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS,
                  0, context_.cuda_stream()>>>(1,
                                              overlaps_pointer,
                                              indices_pointer,
                                              num_total,
                                              nms_,
                                              dpi_,
                                              middle_pointer,
                                              mindex_pointer,
                                              mcounter_pointer);

  // get the counter value
  int num_reduced;
  context_.Copy<int, CUDAContext, CPUContext>(1, mcounter_pointer, &num_reduced);

  // then only copy the valid results
  auto* out_boxes = Output(0);
  out_boxes->Resize(num_reduced, 6);
  float* out_boxes_pointer = out_boxes->mutable_data<float>();

  NMSReduceBoxes<<<CAFFE_GET_BLOCKS(num_reduced), CAFFE_CUDA_NUM_THREADS,
                  0, context_.cuda_stream()>>>(num_reduced,
                                               middle_pointer,
                                               mindex_pointer,
                                               out_boxes_pointer);

  auto* out_feats = Output(1);
  out_feats->Resize(num_reduced, num_feat);
  float* out_feats_pointer = out_feats->mutable_data<float>();

  const int num_reduced_feats = num_feat * num_reduced;
  NMSReduceFeats<<<CAFFE_GET_BLOCKS(num_reduced_feats), CAFFE_CUDA_NUM_THREADS,
                  0, context_.cuda_stream()>>>(num_reduced_feats,
                                               feats_pointer,
                                               mindex_pointer,
                                               num_feat,
                                               out_feats_pointer);

  return true;
}

REGISTER_CUDA_OPERATOR(NMS,
                       NMSOp<float, CUDAContext>);
} // namespace caffe2