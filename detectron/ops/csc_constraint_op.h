#ifndef CAFFE2_OPERATORS_CSC_CONSTRAINT_OP_H_
#define CAFFE2_OPERATORS_CSC_CONSTRAINT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class CSCConstraintOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CSCConstraintOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        polar_(this->template GetSingleArgument<int32_t>("polar", true)) {}
  ~CSCConstraintOp() {}

  bool RunOnDevice() override;

 protected:
  bool polar_;
};

template <typename T, class Context>
class CSCConstraintGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CSCConstraintGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        polar_(this->template GetSingleArgument<int32_t>("polar", true)) {}
  ~CSCConstraintGradientOp() {}

  bool RunOnDevice() override;

 protected:
  bool polar_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CSC_CONSTRAINT_OP_H_
