#include <functional>

#include "csc_m_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(CSCM)
    .NumInputs(4)
    .NumOutputs(3)
    .SetDoc(R"DOC(
)DOC")
    .Arg("tau", "(float) default to 0.7")
    .Arg("max_iter", "(int) default to 0")
    .Arg("debug_info", "(bool) default to false")
    .Arg("fg_threshold", "(float) default to 0.1")
    .Arg("mass_threshold", "(float) default to 0.2")
    .Arg("density_threshold", "(float) default to 0.0")
    .Arg("area_sqrt", "(bool) default to true")
    .Arg("context_scale", "(float) default to 1.8")
    .Input(0, "M", "input tensor of size (BxCxHxW)")
    .Input(1, "X", "input tensor of size (BxC)")
    .Input(2, "Y", "input tensor of size (BxC)")
    .Input(3, "R", "input tensor of size (Nx5)")
    .Output(0, "W", "output tensor of size (NxC)")
    .Output(1, "PL", "output tensor of size (BxC)")
    .Output(2, "NL", "output tensor of size (BxC)");

namespace {

NO_GRADIENT(CSCM);

}  // namespace

}  // namespace caffe2
