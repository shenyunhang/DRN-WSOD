#ifndef CAFFE2_OPERATORS_CENTER_LOSS_OP_H_
#define CAFFE2_OPERATORS_CENTER_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class CenterLossOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CenterLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        top_k_(this->template GetSingleArgument<int32_t>("top_k", 10)),
        update_(this->template GetSingleArgument<int32_t>("update", 128)),
        lr_(this->template GetSingleArgument<float>("lr", 0.5)),
        display_(this->template GetSingleArgument<int32_t>("display", 1280)),
        max_iter_(this->template GetSingleArgument<int32_t>("max_iter", 0)),
        cur_iter_(0),
        ignore_label_(
            this->template GetSingleArgument<int32_t>("ignore_label", -1)),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)),
        init_(true) {}
  ~CenterLossOp() {}

  bool RunOnDevice() override;

 protected:
  int top_k_;
  int update_;
  float lr_;

  int display_;
  int max_iter_;
  int cur_iter_;

  int ignore_label_;

  bool debug_info_;

  vector<set<int> > roi_sets_;
  vector<vector<int> > accum_update_class_;
  float accum_loss_;
  bool init_;
};

template <typename T, class Context>
class CenterLossGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CenterLossGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        top_k_(this->template GetSingleArgument<int32_t>("top_k", 10)),
        update_(this->template GetSingleArgument<int32_t>("update", 128)),
        lr_(this->template GetSingleArgument<float>("lr", 0.5)),
        display_(this->template GetSingleArgument<int32_t>("display", 1280)),
        max_iter_(this->template GetSingleArgument<int32_t>("max_iter", 0)),
        cur_iter_(0),
        ignore_label_(
            this->template GetSingleArgument<int32_t>("ignore_label", -1)),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)),
        init_(true) {}
  ~CenterLossGradientOp() {}

  bool RunOnDevice() override;

 protected:
  int top_k_;
  int update_;
  float lr_;

  int display_;
  int max_iter_;
  int cur_iter_;

  int ignore_label_;

  bool debug_info_;

  vector<set<int> > roi_sets_;
  bool init_;

  Tensor dCF_{Context::GetDeviceType()};
  Tensor ndCF_{Context::GetDeviceType()};
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CENTER_LOSS_OP_H_
