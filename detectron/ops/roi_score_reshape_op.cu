#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "roi_score_reshape_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void forward_kernel(const int nthreads, const T* Xdata,
                               const T* Rdata, const int N,
                               const int num_classes, const int rois_size,
                               T* Ydata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int b = -1;
    int r = 0;
    for (int n = 0; n < N; n++) {
      if (b != Rdata[n * 5 + 0]) {
        r = 0;
        b = Rdata[n * 5 + 0];
      }
      for (int c = 0; c < num_classes; c++) {
        int Xidx = n * num_classes + c;
        int Yidx = ((b * num_classes) + c) * rois_size + r;
        if (Xidx == index) {
          Ydata[Yidx] = Xdata[Xidx];
        }
      }
      r++;
    }
  }
}

template <typename T>
__global__ void backward_kernel(const int nthreads, const T* Rdata,
                                const T* dYdata, const int N,
                                const int num_classes, const int rois_size,
                                T* dXdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int b = -1;
    int r = 0;
    for (int n = 0; n < N; n++) {
      if (b != Rdata[n * 5 + 0]) {
        r = 0;
        b = Rdata[n * 5 + 0];
      }
      for (int c = 0; c < num_classes; c++) {
        int dXidx = n * num_classes + c;
        int dYidx = ((b * num_classes) + c) * rois_size + r;
        if (dXidx == index) {
          dXdata[dXidx] = dYdata[dYidx];
        }
      }
      r++;
    }
  }
}

}  // namespace

// Implementation for the CUDA context.
template <>
bool RoIScoreReshapeOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& R = Input(1);
  auto* Y = Output(0);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(X.dim32(0), R.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), num_classes_);

  CAFFE_ENFORCE_EQ(R.dim(), 2);
  CAFFE_ENFORCE_EQ(R.dim32(0), X.dim32(0));
  CAFFE_ENFORCE_EQ(R.dim32(1), 5);

  Y->Resize(batch_size_, num_classes_, rois_size_, 1);
  math::Set<float, CUDAContext>(Y->numel(), 0.f, Y->mutable_data<float>(),
                                &context_);

  const int N = X.dim32(0);
  const float* Xdata = X.data<float>();
  const float* Rdata = R.data<float>();
  float* Ydata = Y->mutable_data<float>();

  const int nthreads = N * num_classes_;
  forward_kernel<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS, 0,
                          context_.cuda_stream()>>>(
      nthreads, X.data<float>(), R.data<float>(), N, num_classes_, rois_size_,
      Y->mutable_data<float>());

  return true;
}

template <>
bool RoIScoreReshapeGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto& R = Input(1);
  auto* dX = Output(0);

  CAFFE_ENFORCE_EQ(dY.dim(), 4);
  CAFFE_ENFORCE_EQ(dY.dim32(0), batch_size_);
  CAFFE_ENFORCE_EQ(dY.dim32(1), num_classes_);
  CAFFE_ENFORCE_EQ(dY.dim32(2), rois_size_);
  CAFFE_ENFORCE_EQ(dY.dim32(3), 1);

  CAFFE_ENFORCE_EQ(R.dim(), 2);
  CAFFE_ENFORCE_EQ(R.dim32(1), 5);

  dX->Resize(R.dim32(0), num_classes_);
  math::Set<float, CUDAContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                                &context_);

  const int N = R.dim32(0);
  const float* Rdata = R.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  const int nthreads = N * num_classes_;
  backward_kernel<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
      nthreads, R.data<float>(), dY.data<float>(), N, num_classes_, rois_size_,
      dX->mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(RoIScoreReshape, RoIScoreReshapeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(RoIScoreReshapeGradient,
                       RoIScoreReshapeGradientOp<float, CUDAContext>);

}  // namespace caffe2
