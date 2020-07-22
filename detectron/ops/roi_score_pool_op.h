#ifndef CAFFE2_OPERATORS_ROI_SCORE_POOL_OP_H_
#define CAFFE2_OPERATORS_ROI_SCORE_POOL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class RoIScorePoolOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RoIScorePoolOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_classes_(
            this->template GetSingleArgument<int32_t>("num_classes", -1)) {}
  ~RoIScorePoolOp() {}

  bool RunOnDevice() override;

 protected:
  int num_classes_;
};

template <typename T, class Context>
class RoIScorePoolGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RoIScorePoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_classes_(
            this->template GetSingleArgument<int32_t>("num_classes", -1)) {}
  ~RoIScorePoolGradientOp() {}

  bool RunOnDevice() override;

 protected:
  int num_classes_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ROI_SCORE_POOL_OP_H_
