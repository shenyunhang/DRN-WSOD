#include <cfloat>
#include <functional>

#include "deeplab_utility_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(DeeplabUtility)
    .NumInputs(3)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Arg("tau_", "(float) default to 0.5")
    .Arg("softmax_", "(bool) default to false")
    .Input(0, "CPG", "Input tensor of size (BxCxHxW)")
    .Input(1, "L", "Input tensor of size (BxC)")
    .Input(2, "P", "Input tensor of size (BxC)")
    .Input(2, "ML", "Input tensor of size (BxCxHxW)");

namespace {

NO_GRADIENT(DeeplabUtility);

}  // namespace

}  // namespace caffe2
