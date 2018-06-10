#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "compute_box_targets_op.h"

namespace caffe2 {

namespace {

__device__ void _ones(const int size, float* target) {
    for (int i=0; i<size; i++) {
        *(target++) = 1.;
    }
}

__device__ void _compute_targets(const float x1A, const float y1A, const float x2A, const float y2A,
                    const float x1B, const float y1B, const float x2B, const float y2B,
                    const int offset, 
                    float* bbox_targets_pointer,
                    float* bbox_inside_weights_pointer,
                    float* bbox_outside_weights_pointer) {
  const float ex_width = x2A - x1A + 1.;
  const float ex_height = y2A - y1A + 1.;
  const float ex_ctr_x = x1A + 0.5 * ex_width;
  const float ex_ctr_y = y1A + 0.5 * ex_height;

  const float gt_width = x2B - x1B + 1.;
  const float gt_height = y2B - y1B + 1.;
  const float gt_ctr_x = x1B + 0.5 * gt_width;
  const float gt_ctr_y = y1B + 0.5 * gt_height;

  // const float wx = 10.;
  // const float wy = 10.;
  // const float ww = 5.;
  // const float wh = 5.;

  const float target_dx = 10. * (gt_ctr_x - ex_ctr_x) / ex_width;
  const float target_dy = 10. * (gt_ctr_y - ex_ctr_y) / ex_height;
  const float target_dw = 5. * log(gt_width / ex_width);
  const float target_dh = 5. * log(gt_height / ex_height);

  bbox_targets_pointer[offset] = target_dx;
  bbox_targets_pointer[offset+1] = target_dy;
  bbox_targets_pointer[offset+2] = target_dw;
  bbox_targets_pointer[offset+3] = target_dh;

  _ones(4, bbox_inside_weights_pointer + offset);
  _ones(4, bbox_outside_weights_pointer + offset);
}

__global__ void GetBoxes(const int nthreads,
                         const float* rois_pointer,
                         const float* gt_boxes_pointer,
                         const int* labels_pointer,
                         const int* targets_pointer,
                         float* bbox_targets_pointer,
                         float* bbox_inside_weights_pointer,
                         float* bbox_outside_weights_pointer,
                         const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int this_label = labels_pointer[index];
    if (this_label > 0) {
      const int gt_id = targets_pointer[index];
      const int Ap = index * 5;
      const int Bp = gt_id * 5;
      _compute_targets(rois_pointer[Ap+1], rois_pointer[Ap+2], 
                      rois_pointer[Ap+3], rois_pointer[Ap+4],
                      gt_boxes_pointer[Bp], gt_boxes_pointer[Bp+1], 
                      gt_boxes_pointer[Bp+2], gt_boxes_pointer[Bp+3],
                      (index * num_classes + this_label) * 4, 
                      bbox_targets_pointer,
                      bbox_inside_weights_pointer,
                      bbox_outside_weights_pointer);
    }
  }
}

} // namespace

template<>
bool ComputeBoxTargetsOp<float, CUDAContext>::RunOnDevice() {
  auto& rois = Input(0);
  auto& gt_boxes = Input(1);
  auto& labels = Input(2);
  auto& targets = Input(3);

  const float* rois_pointer = rois.data<float>();
  const float* gt_boxes_pointer = gt_boxes.data<float>();
  const int* labels_pointer = labels.data<int>();
  const int* targets_pointer = targets.data<int>();

  const int this_cnt = rois.dim32(0);

  auto* bbox_targets = Output(0);
  auto* bbox_inside_weights = Output(1);
  auto* bbox_outside_weights = Output(2);

  bbox_targets->Resize(this_cnt, 4 * num_classes_);
  bbox_inside_weights->Resize(this_cnt, 4 * num_classes_);
  bbox_outside_weights->Resize(this_cnt, 4 * num_classes_);

  float* bbox_targets_pointer = bbox_targets->mutable_data<float>();
  float* bbox_inside_weights_pointer = bbox_inside_weights->mutable_data<float>();
  float* bbox_outside_weights_pointer = bbox_outside_weights->mutable_data<float>();

  // set to zeros
  const int total_cnt = this_cnt * (4 * num_classes_);
  math::Set<float, CUDAContext>(total_cnt, 0., bbox_targets_pointer, &context_);
  math::Set<float, CUDAContext>(total_cnt, 0., bbox_inside_weights_pointer, &context_);
  math::Set<float, CUDAContext>(total_cnt, 0., bbox_outside_weights_pointer, &context_);

  GetBoxes<<<CAFFE_GET_BLOCKS(this_cnt), CAFFE_CUDA_NUM_THREADS,
            0, context_.cuda_stream()>>>(this_cnt, rois_pointer, gt_boxes_pointer,
                                        labels_pointer, targets_pointer,
                                        bbox_targets_pointer, 
                                        bbox_inside_weights_pointer, 
                                        bbox_outside_weights_pointer,
                                        num_classes_);

  return true;
}

REGISTER_CUDA_OPERATOR(ComputeBoxTargets,
                       ComputeBoxTargetsOp<float, CUDAContext>);
} // namespace caffe2