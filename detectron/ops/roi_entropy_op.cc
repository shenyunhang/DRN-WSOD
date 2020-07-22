#include <functional>

#include "roi_entropy_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(RoIEntropy)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
)DOC")
    .Arg("num_classes", "(int) default to 20")
    .Arg("rm_bg", "(bool) default to true")
    .Arg("debug_info", "(bool) default to false")
    .Input(0, "scores", "input tensor of size (n)")
    .Input(1, "classes", "input tensor of size (n)")
    .Output(0, "entropy", "output tensor of size (c)");

namespace {

NO_GRADIENT(RoIEntropy);

}  // namespace

}  // namespace caffe2
