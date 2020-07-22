#include <functional>

#include "cpg_scale_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(CPGScale)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
)DOC")
    .Arg("tau", "(float) default to 0.7")
    .Input(0, "M", "input tensor of size (BxCxHxW)")
    .Input(1, "X", "input tensor of size (BxC)")
    .Input(2, "Y", "input tensor of size (BxC)")
    .Output(0, "SM", "output tensor of size (BxCxHxW)");

namespace {

NO_GRADIENT(CPGScale);

}  // namespace

}  // namespace caffe2
