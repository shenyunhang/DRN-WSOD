#include "caffe2/operators/box_with_nms_limit_op.h"

#include "caffe2/operators/operator_fallback_gpu.h"

namespace caffe2 {

namespace {}  // namespace

namespace {

REGISTER_CUDA_OPERATOR(BoxWithNMSLimit, GPUFallbackOp);

}  // namespace
}  // namespace caffe2
