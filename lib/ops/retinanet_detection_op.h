#ifndef RETINANET_DETECTION_OP_H_
#define RETINANET_DETECTION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class RetinanetDetectionOp final : public Operator<Context> {
 public:
  RetinanetDetectionOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        k_min_(OperatorBase::GetSingleArgument<int>("k_min", 3)),
        k_max_(OperatorBase::GetSingleArgument<int>("k_max", 7)),
        A_(OperatorBase::GetSingleArgument<int>("A", 9)),
        top_n_(OperatorBase::GetSingleArgument<int>("top_n", 1000)),
        nms_(OperatorBase::GetSingleArgument<float>("nms", .5)),
        thresh_(OperatorBase::GetSingleArgument<float>("thresh", .05)),
        dpi_(OperatorBase::GetSingleArgument<int>("dpi", 100)) {
    DCHECK_GE(k_min_, 1);
    DCHECK_GE(k_max_, 1);
    DCHECK_GE(A_, 1);
    DCHECK_GE(top_n_, 1);
    DCHECK_GE(nms_, 0.);
    DCHECK_LE(nms_, 1.);
    DCHECK_GE(thresh_, 0.);
    DCHECK_LE(thresh_, 1.);
    DCHECK_GE(dpi_, 1); 
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
  int _detection_for_one_image(vector<const float*>& cls_preds,
                              vector<const float*>& cls_probs,
                              vector<const float*>& box_preds,
                              vector<const float*>& anchors,
                              const float* im_info,
                              vector<int>& heights,
                              vector<int>& widths,
                              const int im,
                              const int num_classes,
                              const int num_levels,
                              float* roi_pointer,
                              float* feat_pointer,
                              const int num_rois);

 protected:
  int k_min_;
  int k_max_;
  int A_;
  int top_n_;
  float nms_;
  float thresh_;
  int dpi_;
};

} // namespace caffe2

#endif // RETINANET_DETECTION_OP_H_
