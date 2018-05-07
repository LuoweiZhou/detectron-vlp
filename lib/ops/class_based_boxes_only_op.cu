#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "class_based_boxes_only_op.h"

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

__global__ void ClassBasedBoxesOnlyForward(const int nthreads,
                                      const float* input_boxes,
                                      const float cls_float,
                                      float* cls_box,
                                      int* counter) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int bp = i * 12;
    const float cls_f = input_boxes[bp+5];
    const float valid = input_boxes[bp+6];
    if (cls_f == cls_float && valid > 0.) {
      // returns the old index
      const int idx = gpu_atomic_add(1, counter);
      const int bbp = idx * 13;

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
      // just copy from now on
      _copy(5, input_boxes + (bp+7), cls_box + (bbp+8));
    }
  }
}

} // namespace

template<>
bool ClassBasedBoxesOnlyOp<float, CUDAContext>::RunOnDevice() {
  auto& stats = Input(0);
  const int num_cls = stats.dim32(0);
  DCHECK_LT(class_, num_cls);
  auto& boxes = Input(1);
  const int num_total = boxes.dim32(0);
  DCHECK_EQ(boxes.dim32(1), 12);

  // get the statistics to cpu for the current class
  int current_cls_count;
  const int* stats_pointer = stats.data<int>();
  context_.Copy<int, CUDAContext, CPUContext>(1, stats_pointer + class_, &current_cls_count);
  // use the stats to initialize class based tensors
  auto* cls_boxes = Output(0);
  cls_boxes->Resize(current_cls_count, 13);
  float* cls_boxes_pointer = cls_boxes->mutable_data<float>();
  counter_pointer_ = counter_.mutable_data<int>();

  if (current_cls_count > 0) {
    // reset counter to zero
    math::Set<int, CUDAContext>(1, 0, counter_pointer_, &context_);

    // now copy things to different classes
    ClassBasedBoxesOnlyForward<<<CAFFE_GET_BLOCKS(num_total), CAFFE_CUDA_NUM_THREADS,
                                 0, context_.cuda_stream()>>>(num_total,
                                                              boxes.data<float>(),
                                                              class_float_,
                                                              cls_boxes_pointer,
                                                              counter_pointer_);
  } // otherwise we do not need to do anything!

  return true;
}

REGISTER_CUDA_OPERATOR(ClassBasedBoxesOnly,
                       ClassBasedBoxesOnlyOp<float, CUDAContext>);
} // namespace caffe2