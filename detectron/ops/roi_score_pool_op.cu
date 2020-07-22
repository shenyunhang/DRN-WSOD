#include <functional>

#include "caffe2/core/context_gpu.h"
#include "roi_score_pool_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void kernel_forward(const int nthreads, const T* Xdata,
                               const int batch_size, const int channels,
                               const int height, const int width,
                               int num_classes, T* Ydata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // int w = index % width;
    // int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int b = index / width / height / channels;

    int c_Y = c % num_classes;
    int index_Y = b * num_classes + c_Y;

    gpu_atomic_add(static_cast<T>(Xdata[index]), Ydata + index_Y);
  }

  // CUDA_1D_KERNEL_LOOP(index, nthreads) {
  // int c = (index / 1 / 1) % num_classes;
  // int b = index / 1 / 1 / num_classes;
  // int index_Y = b * num_classes + c;

  // for (int cc = c; cc < channels; cc += num_classes) {
  // for (int h = 0; h < height; h++) {
  // for (int w = 0; w < width; w++) {
  // int index_X = ((b * channels + cc) * height + h) * width + w;
  // Ydata[index_Y] += Xdata[index_X];
  //}
  //}
  //}
  //}
}

template <typename T>
__global__ void kernel_show(const T* Xdata, const int batch_size,
                            const int channels, const int height,
                            const int width, const int ndim) {
  printf("ndim=%d b = %d c = %d h = %d w = %d\n", ndim, batch_size, channels,
         height, width);
  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index_X = ((b * channels + c) * height + h) * width + w;
          printf("b = %d c = %d h = %d w = %d %f\n", b, c, h, w,
                 Xdata[index_X]);
        }
      }
    }
  }
}

template <typename T>
__global__ void kernel_backward(const int nthreads, const T* dYdata,
                                const int batch_size, const int channels,
                                const int height, const int width,
                                int num_classes, T* dXdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // int w = index % width;
    // int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int b = index / width / height / channels;

    int c_Y = c % num_classes;
    int index_Y = b * num_classes + c_Y;

    dXdata[index] = dYdata[index_Y];
  }
}

}  // namespace

template <>
bool RoIScorePoolOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const int batch_size = X.dim32(0);
  auto* Y = Output(0);
  Y->Resize(batch_size, num_classes_);
  math::Set<float, CUDAContext>(Y->numel(), 0.f, Y->mutable_data<float>(),
                                &context_);

  for (int i = 0; i < InputSize(); ++i) {
    const auto& X = Input(i);
    const int channels = X.dim32(1);
    int height, width;
    if (X.dim() == 2) {
      height = 1;
      width = 1;
    } else if (X.dim() == 3) {
      height = X.dim32(2);
      width = 1;
    } else if (X.dim() == 4) {
      height = X.dim32(2);
      width = X.dim32(3);
    }
    const int nthreads = X.numel();

    // kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0,
    // context_.cuda_stream()>>>(
    // X.data<float>(), batch_size, channels, height, width, X.dim());

    kernel_forward<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
        nthreads, X.data<float>(), batch_size, channels, height, width,
        num_classes_, Y->mutable_data<float>());

    // kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0,
    // context_.cuda_stream()>>>(
    // Y->data<float>(), batch_size, num_classes_, 1, 1, Y->dim());
  }
  return true;
}

template <>
bool RoIScorePoolGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const float* dYdata = dY.data<float>();

  for (int i = 1; i < InputSize(); ++i) {
    const auto& X = Input(i);
    const int batch_size = X.dim32(0);
    const int channels = X.dim32(1);
    int height, width;
    if (X.dim() == 2) {
      height = 1;
      width = 1;
    } else if (X.dim() == 3) {
      height = X.dim32(2);
      width = 1;
    } else if (X.dim() == 4) {
      height = X.dim32(2);
      width = X.dim32(3);
    }
    const int nthreads = X.numel();
    auto* dX = Output(i - 1);
    dX->ResizeLike(X);
    math::Set<float, CUDAContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                                  &context_);

    kernel_backward<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,
                             0, context_.cuda_stream()>>>(
        nthreads, dY.data<float>(), batch_size, channels, height, width,
        num_classes_, dX->mutable_data<float>());
  }
  return true;
}

REGISTER_CUDA_OPERATOR(RoIScorePool, RoIScorePoolOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(RoIScorePoolGradient,
                       RoIScorePoolGradientOp<float, CUDAContext>);

}  // namespace caffe2
