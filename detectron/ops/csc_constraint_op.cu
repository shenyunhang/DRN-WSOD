#include <functional>

#include "caffe2/core/context_gpu.h"
#include "csc_constraint_op.h"

namespace caffe2 {

namespace {

template <typename Dtype>
__global__ void min_is_0_kernel(const int nthreads, const Dtype* const x,
                                Dtype* const y) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (x[index] < 0.0)
      y[index] = 0.0;
    else
      y[index] = x[index];
  }
}

template <typename Dtype>
__global__ void max_is_0_kernel(const int nthreads, const Dtype* const x,
                                Dtype* const y) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (x[index] > 0.0)
      y[index] = 0.0;
    else
      y[index] = x[index];
  }
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

}  // namespace

template <>
bool CSCConstraintOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& W = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(W.dim(), 2);
  CAFFE_ENFORCE_EQ(X.dim32(0), W.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), W.dim32(1));

  auto* Y = Output(0);
  Y->ResizeLike(X);

  auto* W_ = Output(1);
  W_->ResizeLike(W);
  W_->CopyFrom(W, false);
  context_.FinishDeviceComputation();

  int nthreads = X.numel();
  if (polar_) {
    // minima is 0
    min_is_0_kernel<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,
                             0, context_.cuda_stream()>>>(
        nthreads, W_->data<float>(), W_->mutable_data<float>());
  } else {
    // maxima is 0
    max_is_0_kernel<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,
                             0, context_.cuda_stream()>>>(
        nthreads, W_->data<float>(), W_->mutable_data<float>());
    math::Scale<float, float, CUDAContext>(
        nthreads, float(-1.0), W_->data<float>(), W_->mutable_data<float>(),
        &context_);
  }

  math::Mul<float, CUDAContext>(nthreads, W_->data<float>(), X.data<float>(),
                                Y->mutable_data<float>(), &context_);

  return true;
}

template <>
bool CSCConstraintGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& W_ = Input(1);

  CAFFE_ENFORCE_EQ(dY.dim(), 2);
  CAFFE_ENFORCE_EQ(W_.dim(), 2);
  CAFFE_ENFORCE_EQ(dY.dim32(0), W_.dim32(0));
  CAFFE_ENFORCE_EQ(dY.dim32(1), W_.dim32(1));

  auto* dX = Output(0);
  dX->ResizeLike(dY);

  int nthreads = dY.numel();
  math::Mul<float, CUDAContext>(nthreads, W_.data<float>(), dY.data<float>(),
                                dX->mutable_data<float>(), &context_);

  return true;
}

REGISTER_CUDA_OPERATOR(CSCConstraint, CSCConstraintOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(CSCConstraintGradient,
                       CSCConstraintGradientOp<float, CUDAContext>);

}  // namespace caffe2
