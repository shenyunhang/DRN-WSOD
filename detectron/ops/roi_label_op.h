#ifndef CAFFE2_OPERATORS_ROI_LABEL_OP_H_
#define CAFFE2_OPERATORS_ROI_LABEL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class RoILabelOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RoILabelOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        cur_iter_(0),
        acc_fg_rois_(0),
        acc_bg_rois_(0),
        acc_fg_weight_(0),
        acc_bg_weight_(0),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)),
        uuid_(this->template GetSingleArgument<int>("uuid", 0)),
        display_(this->template GetSingleArgument<int32_t>("display", 1280)),
        fg_thresh_(this->template GetSingleArgument<float>("fg_thresh", 0.5)),
        bg_thresh_hi_(
            this->template GetSingleArgument<float>("bg_thresh_hi", 0.5)),
        bg_thresh_lo_(
            this->template GetSingleArgument<float>("bg_thresh_lo", -1.0)),
        num_pos_(this->template GetSingleArgument<int>("num_pos", 9999)),
        num_neg_(this->template GetSingleArgument<int>("num_neg", 9999)),
        top_k_(this->template GetSingleArgument<int>("top_k", 1)) {}
  ~RoILabelOp() {}

  bool RunOnDevice() override;

 protected:
  float fg_thresh_;
  float bg_thresh_hi_;
  float bg_thresh_lo_;

  int num_pos_;
  int num_neg_;

  int top_k_;

  bool debug_info_;
  int uuid_;
  int display_;
  int cur_iter_;
  int acc_fg_rois_;
  int acc_bg_rois_;
  T acc_fg_weight_;
  T acc_bg_weight_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ROI_LABEL_OP_H_
