#ifndef GENERATE_PROPOSAL_LABELS_ROIS_ONLY_OP_H_
#define GENERATE_PROPOSAL_LABELS_ROIS_ONLY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class GenerateProposalLabelsRoIsOnlyOp final : public Operator<Context> {
 public:
  GenerateProposalLabelsRoIsOnlyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        rois_per_image_(OperatorBase::GetSingleArgument<int>("rois_per_image", 64)),
        fg_rois_per_image_(OperatorBase::GetSingleArgument<int>("fg_rois_per_image", 16)),
        fg_thresh_(OperatorBase::GetSingleArgument<float>("fg_thresh", 0.5)),
        bg_thresh_hi_(OperatorBase::GetSingleArgument<float>("bg_thresh_hi", 0.5)), 
        bg_thresh_lo_(OperatorBase::GetSingleArgument<float>("bg_thresh_lo", 0.0)),
        im_(OperatorBase::GetSingleArgument<int>("im", 0)),
        rng_seed_(OperatorBase::GetSingleArgument<int>("rng_seed", 3)) {
    DCHECK_GT(rois_per_image_, 0);
    DCHECK_GT(fg_rois_per_image_, 0);
    DCHECK_GE(fg_thresh_, 0.);
    DCHECK_LE(fg_thresh_, 1.);
    DCHECK_GE(bg_thresh_hi_, 0.);
    DCHECK_LE(bg_thresh_hi_, 1.);
    DCHECK_GE(bg_thresh_lo_, 0.);
    DCHECK_LE(bg_thresh_lo_, 1.);
    DCHECK_GE(im_, 0);
    DCHECK_GE(rng_seed_, 0);
    rng_ = std::mt19937(rng_seed_);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int rois_per_image_;
  int fg_rois_per_image_;
  float fg_thresh_;
  float bg_thresh_hi_;
  float bg_thresh_lo_;
  int im_;
  int rng_seed_;
  std::mt19937 rng_;
};

} // namespace caffe2

#endif // GENERATE_PROPOSAL_LABELS_ROIS_ONLY_OP_H_
