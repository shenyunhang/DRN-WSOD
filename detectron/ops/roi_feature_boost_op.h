#ifndef CAFFE2_OPERATORS_ROI_FEATURE_BOOST_OP_H_
#define CAFFE2_OPERATORS_ROI_FEATURE_BOOST_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class RoIFeatureBoostOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RoIFeatureBoostOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~RoIFeatureBoostOp() {}

  bool RunOnDevice() override;

 protected:
};

template <typename T, class Context>
class RoIFeatureBoostGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RoIFeatureBoostGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~RoIFeatureBoostGradientOp() {}

  bool RunOnDevice() override;

 protected:
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ROI_FEATURE_BOOST_OP_H_
