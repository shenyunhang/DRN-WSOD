#include <functional>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "roi_merge_op.h"

namespace caffe2 {

namespace {}  // namespace

REGISTER_CUDA_OPERATOR(RoIMerge, GPUFallbackOp);
REGISTER_CUDA_OPERATOR(RoIMergeGradient, GPUFallbackOp);

}  // namespace caffe2
