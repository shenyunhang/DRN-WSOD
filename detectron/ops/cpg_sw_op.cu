#include <functional>

#include "caffe2/core/context_gpu.h"
#include "cpg_sw_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SigmoidCUDAKernel(const int N, const T* X, T* Y);

template <>
__global__ void SigmoidCUDAKernel<float>(const int N, const float* X,
                                         float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    Y[i] = 1.0f / (1.0f + expf(-__ldg(X + i)));
#else
    Y[i] = 1.0f / (1.0f + expf(-X[i]));
#endif
  }
}

}  // namespace

template <>
bool CPGSWOp<float, CUDAContext>::RunOnDevice() {
  const auto& C = Input(0);
  const auto& M = Input(1);
  const auto& LO = Input(2);
  const auto& LA = Input(3);
  const auto& P = Input(4);
  CAFFE_ENFORCE_EQ(C.dim(), 4);
  CAFFE_ENFORCE_EQ(M.dim(), 4);
  CAFFE_ENFORCE_EQ(LO.dim(), 2);
  CAFFE_ENFORCE_EQ(LA.dim(), 2);
  CAFFE_ENFORCE_EQ(P.dim(), 2);
  CAFFE_ENFORCE_EQ(C.dim32(0), M.dim32(0));
  CAFFE_ENFORCE_EQ(C.dim32(0), LO.dim32(0));
  CAFFE_ENFORCE_EQ(C.dim32(0), LA.dim32(0));
  CAFFE_ENFORCE_EQ(C.dim32(0), P.dim32(0));
  CAFFE_ENFORCE_EQ(C.dim32(1), M.dim32(1));
  CAFFE_ENFORCE_EQ(C.dim32(1), LO.dim32(1));
  CAFFE_ENFORCE_EQ(C.dim32(1), LA.dim32(1));
  CAFFE_ENFORCE_EQ(C.dim32(1), P.dim32(1));
  CAFFE_ENFORCE_EQ(C.dim32(2), M.dim32(2));
  CAFFE_ENFORCE_EQ(C.dim32(3), M.dim32(3));

  const int batch_size = C.dim32(0);
  const int num_classes = C.dim32(1);
  const int height = C.dim32(2);
  const int width = C.dim32(3);

  auto* CO = Output(0);
  CO->CopyFrom(C, false);
  context_.FinishDeviceComputation();

  if (cur_iter_ >= max_iter_) {
    return true;
  }

  Tensor LOcpu = Tensor(LO, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* LOcpudata = LOcpu.data<float>();

  Tensor LAcpu = Tensor(LA, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* LAcpudata = LAcpu.data<float>();

  Tensor Pcpu = Tensor(P, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Pcpudata = Pcpu.data<float>();

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < num_classes; c++) {
      int label_idx = b * num_classes + c;
      float loss_value = LOcpudata[label_idx];
      float label_value = LAcpudata[label_idx];
      float pred_value = Pcpudata[label_idx];

      if (label_value < 0.5) {
        continue;
      }

      if (pred_value < tau_) {
        continue;
      }

      acm_loss_ += loss_value;
      acm_cnt_ += 1;

      if (loss_value > min_loss_) {
        continue;
      }

      acm_sw_loss_ += loss_value;
      acm_sw_cnt_ += 1;

      SigmoidCUDAKernel<float>
          <<<CAFFE_GET_BLOCKS(height * width), CAFFE_CUDA_NUM_THREADS, 0,
             context_.cuda_stream()>>>(
              height * width, M.data<float>() + label_idx * height * width,
              CO->mutable_data<float>() + label_idx * height * width);
    }
  }

  cur_iter_++;

  if (cur_iter_ % 1280 == 0) {
    printf("CPG_SW acm_loss: %f acm_cnt: %d  mean loss: %f\n", acm_loss_,
           acm_cnt_, acm_loss_ / acm_cnt_);
    printf("CPG_SW acm_sw_loss: %f acm_sw_cnt: %d  mean loss: %f\n",
           acm_sw_loss_, acm_sw_cnt_, acm_sw_loss_ / acm_sw_cnt_);
    acm_loss_ = 0;
    acm_cnt_ = 0;
    acm_sw_loss_ = 0;
    acm_sw_cnt_ = 0;
  }

  return true;
}

REGISTER_CUDA_OPERATOR(CPGSW, CPGSWOp<float, CUDAContext>);

}  // namespace caffe2
