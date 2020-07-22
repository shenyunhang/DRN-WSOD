#include <functional>

#include "cpg_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(CPG)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
)DOC")
    .Arg("tau", "(float) default to 0.7")
    .Arg("max_iter", "(int) default to 0")
    .Arg("debug_info", "(bool) default to false")
    .Arg("cpg_net_name", "(string) default to cpg")
    .Arg("pred_blob_name", "(string) default to cls_prob")
    .Arg("data_blob_name", "(string) default to data")
    .Input(0, "X", "input tensor of size (BxC)")
    .Input(1, "Y", "input tensor of size (BxC)")
    .Output(0, "M", "output tensor of size (BxCxHxW)");

namespace {

NO_GRADIENT(CPG);

}  // namespace

}  // namespace caffe2
