#include <functional>

#include "roi_iou_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(RoIIoU)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
)DOC")
    .Arg("debug_info", "(bool) default to false")
    .Input(0, "rois", "input tensor of size (n x 5)")
    .Output(0, "iou", "output tensor of size (n x n)");

namespace {

NO_GRADIENT(RoIIoU);

}  // namespace

}  // namespace caffe2
