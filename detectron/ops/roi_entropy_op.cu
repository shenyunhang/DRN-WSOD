#include <functional>

#include "caffe2/core/context_gpu.h"
#include "roi_entropy_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void Add_A_not_1(const int nthreads, const T* A, const T* B, T* C) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    if (A[idx] == 1.f) {
      continue;
    }
    C[idx] = A[idx] + B[idx];
  }
}

template <typename T>
__global__ void count(const int nthreads, const T* Sdata, const T* Cdata,
                      const int c_offset, T* Ndata, T* CSdata) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int c = Cdata[idx] + c_offset;

    T* address = Ndata + c;
    T val = 1.0;
    gpu_atomic_add(val, address);

    address = CSdata + c;
    val = Sdata[idx];
    gpu_atomic_add(val, address);
  }
}

template <typename T>
__global__ void entropy(const int nthreads, const T* Sdata, const T* Cdata,
                        const T* Ndata, const T* CSdata, const int c_offset,
                        T* Edata) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int c = Cdata[idx] + c_offset;

    T p = Sdata[idx] / CSdata[c];
    T* address = Edata + c;
    T val = -1.0 * -1.0 * p * log(p);
    if (Ndata[c] == 1) {
      val = 0.0;
    } else {
      val = val / log(Ndata[c]);
    }
    gpu_atomic_add(val, address);
  }
}

}  // namespace

/*
$1 - \frac{- \sum{p_i \ln{p_i}}}{ln(n)}$
*/

template <>
bool RoIEntropyOp<float, CUDAContext>::RunOnDevice() {
  const auto& S = Input(0);
  const auto& C = Input(1);

  CAFFE_ENFORCE_EQ(S.ndim(), 1);
  CAFFE_ENFORCE_EQ(C.ndim(), 1);
  CAFFE_ENFORCE_EQ(S.dim32(0), C.dim32(0));

  const int n = S.dim32(0);

  auto* E = Output(0);

  E->Resize(1, num_classes_);
  math::Set<float, CUDAContext>(E->size(), 1.f, E->mutable_data<float>(),
                                &context_);

  Tensor N = Tensor(caffe2::CUDA);
  Tensor CS = Tensor(caffe2::CUDA);
  N.ResizeLike(*E);
  CS.ResizeLike(*E);
  math::Set<float, CUDAContext>(N.size(), 0.f, N.mutable_data<float>(),
                                &context_);
  math::Set<float, CUDAContext>(CS.size(), 0.f, CS.mutable_data<float>(),
                                &context_);

  if (init_) {
    mean_.ResizeLike(*E);
    math::Set<float, CUDAContext>(mean_.size(), 0.f,
                                  mean_.mutable_data<float>(), &context_);
    init_ = false;
  }

  int c_offset = 0;
  if (rm_bg_) {
    c_offset = -1;
  }

  count<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0,
                 context_.cuda_stream()>>>(n, S.data<float>(), C.data<float>(),
                                           c_offset, N.mutable_data<float>(),
                                           CS.mutable_data<float>());

  entropy<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0,
                   context_.cuda_stream()>>>(
      n, S.data<float>(), C.data<float>(), N.data<float>(), CS.data<float>(),
      c_offset, E->mutable_data<float>());

  Add_A_not_1<float><<<CAFFE_GET_BLOCKS(mean_.size()), CAFFE_CUDA_NUM_THREADS,
                       0, context_.cuda_stream()>>>(
      mean_.size(), E->data<float>(), mean_.data<float>(),
      mean_.mutable_data<float>());

  cur_iter_++;
  if (cur_iter_ % display_ == 0 || cur_iter_ == 1) {
    Tensor mean_cpu = Tensor(caffe2::CPU);
    mean_cpu.ResizeLike(mean_);

    mean_cpu.CopyFrom(mean_, false);
    context_.FinishDeviceComputation();

    const float* mean_cpu_data = mean_cpu.data<float>();

    std::cout << "RoIEntropy #iter_: " << cur_iter_ << std::endl;
    for (int i = 0; i < mean_cpu.size(); i++) {
      std::cout << "  " << mean_cpu_data[i];
    }
    std::cout << std::endl;

    init_ = true;
  }

  return true;
}

REGISTER_CUDA_OPERATOR(RoIEntropy, RoIEntropyOp<float, CUDAContext>);

}  // namespace caffe2
