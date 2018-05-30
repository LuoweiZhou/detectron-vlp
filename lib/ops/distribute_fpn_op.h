#ifndef DISTRIBUTE_FPN_OP_H_
#define DISTRIBUTE_FPN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class DistributeFPNOp final : public Operator<Context> {
 public:
  DistributeFPNOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
      k_min_(OperatorBase::GetSingleArgument<int>("k_min", 2)),
      k_max_(OperatorBase::GetSingleArgument<int>("k_max", 5)),
      c_scale_(OperatorBase::GetSingleArgument<int>("c_scale", 224)),
      c_level_(OperatorBase::GetSingleArgument<int>("c_level", 4)) {
        DCHECK_GE(k_min_, 1);
        DCHECK_GE(k_max_, 1);
        DCHECK_GE(k_max_, k_min_);
        num_levels_ = k_max_ - k_min_ + 1;
        stats_.Resize(num_levels_);
        counters_.Resize(num_levels_);
        accs_.Resize(num_levels_);
        c_scale_f_ = static_cast<float>(c_scale_);
        c_level_f_ = static_cast<float>(c_level_);
      }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int k_min_;
  int k_max_;
  int num_levels_;
  int c_scale_;
  int c_level_;
  float c_scale_f_;
  float c_level_f_;

  Tensor<Context> stats_;
  Tensor<Context> counters_;
  Tensor<Context> accs_;
  Tensor<Context> levels_;
};

} // namespace caffe2

#endif // DISTRIBUTE_FPN_OP_H_
