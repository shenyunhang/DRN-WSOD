#ifndef CAFFE2_OPERATORS_ROI_MERGE_OP_H_
#define CAFFE2_OPERATORS_ROI_MERGE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class RoIMergeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RoIMergeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        cur_iter_(0),
        acc_num_top_id_(0),
        acc_max_clique_(0),
        acc_min_clique_(0),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)),
        max_epoch_(this->template GetSingleArgument<int32_t>("max_epoch", 20)),
        size_epoch_(
            this->template GetSingleArgument<int32_t>("size_epoch", 5000)),
        display_(this->template GetSingleArgument<int32_t>("display", 1280)) {}
  ~RoIMergeOp() {}

  bool RunOnDevice() override;

 protected:
  bool debug_info_;
  int display_;

  int cur_iter_;
  int max_epoch_;
  int size_epoch_;

  int acc_num_top_id_;
  int acc_max_clique_;
  int acc_min_clique_;
};

template <typename T, class Context>
class RoIMergeGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RoIMergeGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~RoIMergeGradientOp() {}

  bool RunOnDevice() override;

 protected:
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ROI_MERGE_OP_H_
