#ifndef GENERATE_PROPOSALS_SINGLE_IMAGE_OP_H_
#define GENERATE_PROPOSALS_SINGLE_IMAGE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class GenerateProposalsSingleImageOp final : public Operator<Context> {
 public:
  GenerateProposalsSingleImageOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        pre_top_n_(OperatorBase::GetSingleArgument<int>("pre_top_n", 12000)),
        post_top_n_(OperatorBase::GetSingleArgument<int>("post_top_n", 2000)),
        nms_(OperatorBase::GetSingleArgument<float>("nms", 0.7)),
        im_(OperatorBase::GetSingleArgument<int>("im", 0)),
        stride_(OperatorBase::GetSingleArgument<int>("stride", 4)) {
    DCHECK_GE(pre_top_n_, 1);
    DCHECK_GE(post_top_n_, 1);
    DCHECK_GE(nms_, 0.);
    DCHECK_LE(nms_, 1.);
    DCHECK_GE(im_, 0);
    DCHECK_GE(stride_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int pre_top_n_;
  int post_top_n_;
  float nms_;
  int im_;
  int stride_;

  Tensor<Context> Yv;
  Tensor<Context> Yi;
  Tensor<Context> rois_raw;
};

} // namespace caffe2

#endif // GENERATE_PROPOSALS_SINGLE_IMAGE_OP_H_
