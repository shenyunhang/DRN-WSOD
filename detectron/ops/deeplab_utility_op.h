#ifndef CAFFE2_OPERATORS_DEEPLAB_UTILITY_OP_H_
#define CAFFE2_OPERATORS_DEEPLAB_UTILITY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class DeeplabUtilityOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  DeeplabUtilityOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        tau_(this->template GetSingleArgument<float>("tau", 0.7)),
        softmax_(this->template GetSingleArgument<bool>("softmax", false)),
        fg_th_(this->template GetSingleArgument<float>("fg_th", 0.1)),
        bg_th_(this->template GetSingleArgument<float>("bg_th", 0.005)) {}
  ~DeeplabUtilityOp() {}

  bool RunOnDevice() override;

 protected:
  float tau_;
  float softmax_;
  float fg_th_;
  float bg_th_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_DEEPLAB_UTILITY_OP_H_
