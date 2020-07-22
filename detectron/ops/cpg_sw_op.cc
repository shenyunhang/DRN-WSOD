#include <functional>

#include "cpg_sw_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(CPGSW)
    .NumInputs(5)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
)DOC")
    .Arg("tau", "(float) default to 0.7")
    .Arg("max_iter", "(int) default to 0")
    .Arg("debug_info", "(bool) default to false")
    .Input(0, "C", "input tensor of size (BxCxHxW)")
    .Input(1, "M", "input tensor of size (BxCxHxW)")
    .Input(2, "LO", "input tensor of size (BxC)")
    .Input(3, "LA", "input tensor of size (BxC)")
    .Input(4, "P", "input tensor of size (BxC)")
    .Output(0, "CO", "output tensor of size (BxCxHxW)");

namespace {

NO_GRADIENT(CPGSW);

}  // namespace

}  // namespace caffe2
