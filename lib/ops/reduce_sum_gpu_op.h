#ifndef REDUCE_SUM_GPU_OP_H_
#define REDUCE_SUM_GPU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class ReduceSumGPUOp final : public Operator<Context> {
 public:
  ReduceSumGPUOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
  
};

} // namespace caffe2

#endif // REDUCE_SUM_GPU_OP_H_
