#include <cfloat>
#include <functional>

#include "caffe2/core/context_gpu.h"
#include "cpg_scale_op.h"

namespace caffe2 {

namespace {

template <typename T>
T get_max(const int N, const T* data) {
  T max_val = -FLT_MAX;
  for (int i = 0; i < N; i++) {
    if (*data > max_val) {
      max_val = *data;
    }
    data += 1;
  }
  return max_val;
}

}  // namespace

template <>
bool CPGScaleOp<float, CUDAContext>::RunOnDevice() {
  const auto& M = Input(0);
  const auto& X = Input(1);
  const auto& Y = Input(2);

  CAFFE_ENFORCE_EQ(M.dim(), 4);
  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(Y.dim(), 2);
  CAFFE_ENFORCE_EQ(M.dim32(0), X.dim32(0));
  CAFFE_ENFORCE_EQ(M.dim32(0), Y.dim32(0));
  CAFFE_ENFORCE_EQ(M.dim32(1), X.dim32(1));
  CAFFE_ENFORCE_EQ(M.dim32(1), Y.dim32(1));

  const int batch_size = M.dim32(0);
  const int num_classes = M.dim32(1);
  const int cpg_height = M.dim32(2);
  const int cpg_width = M.dim32(3);

  auto* SM = Output(0);

  SM->ResizeLike(M);
  math::Set<float, CUDAContext>(SM->numel(), 0.f, SM->mutable_data<float>(),
                                &context_);
  // SM->CopyFrom<CUDAContext>(M, false);
  // context_.FinishDeviceComputation();

  Tensor Xcpu = Tensor(X, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Xcpudata = Xcpu.data<float>();

  Tensor Ycpu = Tensor(Y, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Ycpudata = Ycpu.data<float>();

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < num_classes; c++) {
      int label_idx = b * num_classes + c;
      float label_value = Xcpudata[label_idx];
      float pred_value = Ycpudata[label_idx];

      if (label_value == 0.0) {
        continue;
      }
      if (pred_value < tau_) {
        continue;
      }

      // Get CPG map
      Tensor m = Tensor(caffe2::CUDA);
      m.Resize(cpg_height, cpg_width);
      math::Abs<float, CUDAContext>(
          m.numel(), M.data<float>() + cpg_height * cpg_width * label_idx,
          m.mutable_data<float>(), &context_);

      // Get max value
      Tensor mcpu = Tensor(m, caffe2::CPU);
      context_.FinishDeviceComputation();
      float max_val = get_max<float>(mcpu.numel(), mcpu.data<float>());

      if (max_val == 1.0 || max_val == 0.0) {
        continue;
      }

      const float scale = 1. / max_val;

      math::Scale<float, float, CUDAContext>(
          m.numel(), float(scale), m.data<float>(),
          SM->mutable_data<float>() + cpg_height * cpg_width * label_idx,
          &context_);
    }
  }

  return true;
}

REGISTER_CUDA_OPERATOR(CPGScale, CPGScaleOp<float, CUDAContext>);

}  // namespace caffe2
