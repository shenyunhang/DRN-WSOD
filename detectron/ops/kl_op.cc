#include "kl_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(KL)
    .Arg("ignore_value", R"DOC(default is 0.5.)DOC")
    .NumInputs(2 + 1)
    .NumOutputs(2)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "p", "matrix for each example and class.")
    .Input(1, "q", "matrix, same shape as p.")
    .Output(0, "divergence", "Vector with the divergence for each example.")
    .Output(1, "count", "");

OPERATOR_SCHEMA(KLGradient).NumInputs(4).NumOutputs(1);

struct GetKLGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    auto ignore_value =
        argsHelper.GetSingleArgument<float>("ignore_value", 0.5);
    return SingleGradientDef(
        "KLGradient", "", vector<string>{GO(0), I(0), I(1), O(1)},
        vector<string>{GI(0)},
        vector<Argument>{MakeArgument<float>("ignore_value", ignore_value)});
  }
};
REGISTER_GRADIENT(KL, GetKLGradient);

}  // namespace caffe2
