#include <functional>

#include "caffe2/core/context_gpu.h"
#include "roi_feature_boost_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void kernel_forward(const int nthreads, const T* Xdata,
                               const T* Sdata, const int batch_size,
                               const int feature_size, T* Ydata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int f = index % feature_size;
    int b = index / feature_size;

    int index_S = b;
    int index_XY = b * feature_size + f;
    Ydata[index_XY] = Xdata[index_XY] * Sdata[index_S];
  }
}

template <typename T>
__global__ void kernel_backward(const int nthreads, const T* dYdata,
                                const T* Sdata, const int batch_size,
                                const int feature_size, T* dXdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int f = index % feature_size;
    int b = index / feature_size;

    int index_S = b;
    int index_dXY = b * feature_size + f;
    dXdata[index_dXY] = dYdata[index_dXY] * Sdata[index_S];
  }
}

}  // namespace

template <>
bool RoIFeatureBoostOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& S = Input(1);

  CAFFE_ENFORCE_EQ(S.dim32(0), S.numel());
  CAFFE_ENFORCE_EQ(X.dim32(0), S.dim32(0));

  const int batch_size = X.dim32(0);
  const int feature_size = X.size_from_dim(1);

  const float* Xdata = X.data<float>();
  const float* Sdata = S.data<float>();

  auto* Y = Output(0);
  Y->ResizeLike(X);
  float* Ydata = Y->mutable_data<float>();

  const int nthreads = X.numel();

  kernel_forward<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0,
                          context_.cuda_stream()>>>(
      nthreads, Xdata, Sdata, batch_size, feature_size, Ydata);

  return true;
}

template <>
bool RoIFeatureBoostGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& S = Input(1);

  CAFFE_ENFORCE_EQ(S.dim32(0), S.numel());
  CAFFE_ENFORCE_EQ(dY.dim32(0), S.dim32(0));

  const int batch_size = dY.dim32(0);
  const int feature_size = dY.size_from_dim(1);

  const float* dYdata = dY.data<float>();
  const float* Sdata = S.data<float>();

  auto* dX = Output(0);
  dX->ResizeLike(dY);
  float* dXdata = dX->mutable_data<float>();

  const int nthreads = dY.numel();

  kernel_backward<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
      nthreads, dYdata, Sdata, batch_size, feature_size, dXdata);
  return true;
}

REGISTER_CUDA_OPERATOR(RoIFeatureBoost, RoIFeatureBoostOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(RoIFeatureBoostGradient,
                       RoIFeatureBoostGradientOp<float, CUDAContext>);

}  // namespace caffe2
