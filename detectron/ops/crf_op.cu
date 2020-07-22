#include <functional>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "crf_op.h"

namespace caffe2 {

namespace {}  // namespace

REGISTER_CUDA_OPERATOR(DenseCRF, GPUFallbackOp);

}  // namespace caffe2
