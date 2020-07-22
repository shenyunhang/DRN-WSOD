#include <cfloat>
#include <functional>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "pcl_loss_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void get_norm_kernel(const int nthreads, const T* Xdata,
                                const T* Ldata, const int N, const int C,
                                const int B, T* norm) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // int n = index / C;
    int c = index % C;
    if (Ldata[c] < 0.5) {
      continue;
    }
    gpu_atomic_add(static_cast<T>(1), norm);
  }
}

template <typename T>
__global__ void Forward(const int nthreads, const T* Xdata, const T* Ldata,
                        const int N, const int C, const int B,
                        const T kLOG_THRESHOLD, T* Ydata, T* norm) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / C;
    int c = index % C;
    if (Ldata[c] < 0.5) {
      continue;
    }
    float prob = max(Xdata[n * C + c], kLOG_THRESHOLD);
    float loss = -prob * log(prob);
    gpu_atomic_add(static_cast<T>(loss), Ydata);
    gpu_atomic_add(static_cast<T>(1), norm);
    // printf("loss: %f %f %f", loss, Ydata[0], norm[0]);
  }
}

template <typename T>
__global__ void Backward(const int nthreads, const T* Xdata, const T* Ldata,
                         const int N, const int C, const int B, const T* scale,
                         const T kLOG_THRESHOLD, const T kDIFF_THRESHOLD,
                         T* dXdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / C;
    int c = index % C;
    if (Ldata[c] < 0.5) {
      continue;
    }
    float prob = max(Xdata[n * C + c], kLOG_THRESHOLD);
    dXdata[index] = min(scale[0] * (-1 + (-1) * log(prob)), kDIFF_THRESHOLD);
  }
}

}  // namespace

template <>
bool PCLLossOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& L = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(L.dim(), 2);
  CAFFE_ENFORCE_EQ(X.dim32(1), L.dim32(1));

  int N = X.dim32(0);
  int C = X.dim32(1);
  int B = L.dim32(0);

  auto* Y = Output(0);
  Y->Resize(vector<int64_t>());
  math::Set<float, CUDAContext>(Y->numel(), 0.f, Y->mutable_data<float>(),
                                &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  auto* Ydata = Y->mutable_data<float>();

  Tensor norm_gpu(caffe2::CUDA);
  norm_gpu.Resize(vector<int64_t>());
  math::Set<float, CUDAContext>(norm_gpu.numel(), 1.f,
                                norm_gpu.mutable_data<float>(), &context_);

  CAFFE_ENFORCE_EQ(L.dim32(0), 1);
  Forward<float><<<CAFFE_GET_BLOCKS(X.numel()), CAFFE_CUDA_NUM_THREADS, 0,
                   context_.cuda_stream()>>>(X.numel(), Xdata, Ldata, N, C, B,
                                             kLOG_THRESHOLD(), Ydata,
                                             norm_gpu.mutable_data<float>());

  math::Div<float, CUDAContext>(1, Y->data<float>(), norm_gpu.data<float>(),
                                Y->mutable_data<float>(), &context_);

  return true;
}

template <>
bool PCLLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& L = Input(1);
  auto& dY = Input(2);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(L.dim(), 2);
  CAFFE_ENFORCE_EQ(X.dim32(1), L.dim32(1));
  CAFFE_ENFORCE_EQ(dY.numel(), 1);

  int N = X.dim32(0);
  int C = X.dim32(1);
  int B = L.dim32(0);

  auto* dX = Output(0);
  dX->ResizeLike(X);
  math::Set<float, CUDAContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                                &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  Tensor scale_gpu(caffe2::CUDA);
  scale_gpu.Resize(vector<int64_t>());
  math::Set<float, CUDAContext>(scale_gpu.numel(), 1.f,
                                scale_gpu.mutable_data<float>(), &context_);

  CAFFE_ENFORCE_EQ(L.dim32(0), 1);
  get_norm_kernel<float><<<CAFFE_GET_BLOCKS(X.numel()), CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
      X.numel(), Xdata, Ldata, N, C, B, scale_gpu.mutable_data<float>());

  math::Div<float, CUDAContext>(1, dYdata, scale_gpu.data<float>(),
                                scale_gpu.mutable_data<float>(), &context_);

  CAFFE_ENFORCE_EQ(L.dim32(0), 1);
  Backward<float><<<CAFFE_GET_BLOCKS(X.numel()), CAFFE_CUDA_NUM_THREADS, 0,
                    context_.cuda_stream()>>>(
      X.numel(), Xdata, Ldata, N, C, B, scale_gpu.data<float>(),
      kLOG_THRESHOLD(), kDIFF_THRESHOLD(), dXdata);

  return true;
}

// REGISTER_CUDA_OPERATOR(PCLLoss, PCLLossOp<float, CUDAContext>);
// REGISTER_CUDA_OPERATOR(PCLLossGradient, PCLLossGradientOp<float,
// CUDAContext>);

REGISTER_CUDA_OPERATOR(PCLLoss, GPUFallbackOp);
REGISTER_CUDA_OPERATOR(PCLLossGradient, GPUFallbackOp);

}  // namespace caffe2
