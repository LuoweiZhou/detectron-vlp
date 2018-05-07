#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "class_based_boxes_op.h"

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

__global__ void ClassBasedBoxesForward(const int nthreads,
                                      const float* input_boxes,
                                      const float* input_feats,
                                      const int num_feat,
                                      float** cls_boxes,
                                      float** cls_feats,
                                      int* counters) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int bp = i * 7;
    if (input_boxes[bp+6] > 0.) {
      const int fp = i * num_feat;
      const float cls_f = input_boxes[bp+5];
      const int cls = static_cast<int>(cls_f);
      // returns the old index
      const int idx = gpu_atomic_add(1, counters + cls);
      const int bbp = idx * 8;
      const int ffp = idx * num_feat;
      float* cls_box = cls_boxes[cls];
      float* cls_feat = cls_feats[cls];
      const float x1 = input_boxes[bp];
      const float y1 = input_boxes[bp+1];
      const float x2 = input_boxes[bp+2];
      const float y2 = input_boxes[bp+3];
      cls_box[bbp] = x1;
      cls_box[bbp+1] = y1;
      cls_box[bbp+2] = x2;
      cls_box[bbp+3] = y2;
      cls_box[bbp+4] = input_boxes[bp+4];
      cls_box[bbp+5] = (x2 - x1 + 1.) * (y2 - y1 + 1.);
      cls_box[bbp+6] = cls_f;
      // leave a dimension to encode suppressed information
      cls_box[bbp+7] = 0.;
      _copy(num_feat, input_feats + fp, cls_feat + ffp);
    }
  }
}

} // namespace

template<>
bool ClassBasedBoxesOp<float, CUDAContext>::RunOnDevice() {
  auto& stats = Input(0);
  const int num_cls = stats.dim32(0);
  auto& boxes = Input(1);
  const int num_total = boxes.dim32(0);
  DCHECK_EQ(boxes.dim32(1), 7);
  auto& feats = Input(2);
  DCHECK_EQ(feats.dim32(0), num_total);
  const int num_feat = feats.dim32(1);
  DCHECK_EQ(OutputSize(), 2 * num_cls);

  // get the statistics to cpu
  int stats_cpu[num_cls];
  const int* stats_pointer = stats.data<int>();
  context_.Copy<int, CUDAContext, CPUContext>(num_cls, stats_pointer, stats_cpu);
  // use the stats to initialize class based tensors
  float* cls_box_pointers[num_cls];
  float* cls_feat_pointers[num_cls];
  for (int i=0; i<num_cls; i++) {
    // x1, y1, x2, y2, score, area, c, zero
    const int current_cls_count = stats_cpu[i];
    Output(i*2)->Resize(current_cls_count, 8);
    cls_box_pointers[i] = Output(i*2)->mutable_data<float>();
    Output(i*2+1)->Resize(current_cls_count, num_feat);
    cls_feat_pointers[i] = Output(i*2+1)->mutable_data<float>();
  }
  // initialize counter
  counter_.Resize(num_cls);
  int* counter_pointers = counter_.mutable_data<int>();
  // reset counters to zero
  math::Set<int, CUDAContext>(num_cls, 0, counter_pointers, &context_);
  // copy to the gpu memory
  // 64 bit machine: 2
  const int virtual_size = num_cls * size_ratio_;
  pointer_boxes_.Resize(virtual_size);
  pointer_feats_.Resize(virtual_size);
  long* boxes_pointers = pointer_boxes_.mutable_data<long>();
  long* feats_pointers = pointer_feats_.mutable_data<long>();
  context_.Copy<long, CPUContext, CUDAContext>(virtual_size, 
                                                reinterpret_cast<long*>(cls_box_pointers), boxes_pointers);
  context_.Copy<long, CPUContext, CUDAContext>(virtual_size, 
                                                reinterpret_cast<long*>(cls_feat_pointers), feats_pointers);

  // now copy things to different classes
  ClassBasedBoxesForward<<<CAFFE_GET_BLOCKS(num_total), CAFFE_CUDA_NUM_THREADS,
                               0, context_.cuda_stream()>>>(num_total,
                                                            boxes.data<float>(),
                                                            feats.data<float>(),
                                                            num_feat,
                                                            reinterpret_cast<float**>(boxes_pointers),
                                                            reinterpret_cast<float**>(feats_pointers),
                                                            counter_pointers);

  return true;
}

REGISTER_CUDA_OPERATOR(ClassBasedBoxes,
                       ClassBasedBoxesOp<float, CUDAContext>);
} // namespace caffe2