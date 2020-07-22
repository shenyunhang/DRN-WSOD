#ifndef CAFFE2_OPERATORS_CPG_SCALE_OP_H_
#define CAFFE2_OPERATORS_CPG_SCALE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class CPGScaleOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CPGScaleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        tau_(this->template GetSingleArgument<float>("tau", 0.7)) {}
  ~CPGScaleOp() {}

  bool RunOnDevice() override;

 protected:
  float tau_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CPG_SCALE_OP_H_
