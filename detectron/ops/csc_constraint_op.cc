#include <functional>

#include "csc_constraint_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(CSCConstraint)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
)DOC")
    .Arg("polar", "(bool) default to true")
    .Output(0, "Y", "output tensor of size (NxC)");

OPERATOR_SCHEMA(CSCConstraintGradient).NumInputs(2).NumOutputs(1);

namespace {

class GetCSCConstraintGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(def_.type() + "Gradient", "",
                             vector<string>{GO(0), O(1)},
                             vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(CSCConstraint, GetCSCConstraintGradient);

}  // namespace

}  // namespace caffe2
