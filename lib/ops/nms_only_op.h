#ifndef NMS_ONLY_OP_H_
#define NMS_ONLY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class NMSOnlyOp final : public Operator<Context> {
 public:
  NMSOnlyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
      nms_(OperatorBase::GetSingleArgument<float>("nms", .5)),
      dpi_(OperatorBase::GetSingleArgument<int>("dpi", 100)) {
        DCHECK_GE(nms_, 0.);
        DCHECK_LE(nms_, 1.);
        DCHECK_GE(dpi_, 1); 
      }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float nms_;
  int dpi_;

  Tensor<Context> values;
  Tensor<Context> indices;
  Tensor<Context> overlaps;
  Tensor<Context> middle;
  Tensor<Context> mindex;
  Tensor<Context> mcounter;
};

} // namespace caffe2

#endif // NMS_ONLY_OP_H_
