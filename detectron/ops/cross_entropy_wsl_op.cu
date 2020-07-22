#include <assert.h>
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "cross_entropy_wsl_op.h"

namespace caffe2 {

namespace {
__global__ void LabelBalanceWSLKernel(const int outer_size,
                                      const int inner_size,
                                      const int* targets_ptr,
                                      const float ignore_value,
                                      float* count_ptr) {
  int i = blockIdx.x;
  int last_idx = (i + 1) * inner_size;
  float pos = 0;
  float neg = 0;
  for (int in_idx = i * inner_size + threadIdx.x; in_idx < last_idx;
       in_idx += blockDim.x) {
    if (targets_ptr[in_idx] == ignore_value) {
      continue;
    }
    if (targets_ptr[in_idx] > 0) {
      pos += 1;
    } else {
      neg += 1;
    }
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float pos_sum = BlockReduce(temp_storage).Sum(pos);

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce2;
  __shared__ typename BlockReduce2::TempStorage temp_storage2;
  float neg_sum = BlockReduce2(temp_storage2).Sum(neg);

  if (threadIdx.x == 0) {
    count_ptr[i * 2] = pos_sum;
    count_ptr[i * 2 + 1] = neg_sum;
  }
}

__global__ void LabelCrossEntropyWSLKernel_BATCHWISE(
    const int outer_size, const int C, const int inner_size, const float* Xdata,
    const int* labeldata, const float* countdata, const float log_threshold,
    const float ignore_value, float* Ydata) {
  // outer_size = B
  // C = C
  // inner_size = H * W
  // Xdata = B * C * H * W
  // labeldata = B * 1 * H * W
  int i = blockIdx.x;
  float pos = countdata[i * 2];
  float neg = countdata[i * 2 + 1];
  float value = 0;
  for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
    int labelidx = i * inner_size + j;
    int Xidx = (i * C + labeldata[labelidx]) * inner_size + j;
    CUDA_KERNEL_ASSERT(labeldata[labelidx] >= 0 && labeldata[labelidx] < C);
    if (labeldata[labelidx] == ignore_value) {
      continue;
    }
    if (labeldata[labelidx] > 0) {
      value += -logf(max(Xdata[Xidx], log_threshold)) / pos;
    } else {
      value += -logf(max(Xdata[Xidx], log_threshold)) / neg;
    }
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float sum = BlockReduce(temp_storage).Sum(value);
  if (threadIdx.x == 0) {
    // Ydata[i] = sum / inner_size;
    Ydata[i] = sum;
  }
}

__global__ void LabelCrossEntropyWSLKernel_CLASSWISE(
    const int outer_size, const int C, const int inner_size, const float* Xdata,
    const int* labeldata, const float* countdata, const int batch_size,
    const int num_classes, const float log_threshold, const float ignore_value,
    float* Ydata) {
  // outer_size = B * C
  // inner_size = H * W
  // Xdata = B * C * H * W
  // labeldata = B * 1 * H * W
  int i = blockIdx.x;
  int b = i / num_classes;
  int c = i % num_classes;
  float pos = countdata[b * 2];
  float neg = countdata[b * 2 + 1];
  float value = 0;
  for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
    int Xidx = i * inner_size + j;
    int labelidx = b * inner_size + j;
    CUDA_KERNEL_ASSERT(labeldata[labelidx] >= 0 && labeldata[labelidx] < C);
    if (labeldata[labelidx] != c) {
      continue;
    }
    if (labeldata[labelidx] == ignore_value) {
      continue;
    }
    if (labeldata[labelidx] > 0) {
      value += -logf(max(Xdata[Xidx], log_threshold)) / pos;
    } else {
      value += -logf(max(Xdata[Xidx], log_threshold)) / neg;
    }
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float sum = BlockReduce(temp_storage).Sum(value);
  if (threadIdx.x == 0) {
    // Ydata[i] = sum / inner_size;
    Ydata[i] = sum;
  }
}

__global__ void LabelCrossEntropyWSLGradientKernel_BATCHWISE(
    const int outer_size, const int C, const int inner_size, const float* Xdata,
    const int* labeldata, const float* dYdata, const float* countdata,
    const float log_threshold, const float ignore_value, float* dXdata) {
  int i = blockIdx.x;
  float pos = countdata[i * 2];
  float neg = countdata[i * 2 + 1];
  for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
    int labelidx = i * inner_size + j;
    int Xidx = (i * C + labeldata[labelidx]) * inner_size + j;
    CUDA_KERNEL_ASSERT(labeldata[labelidx] >= 0 && labeldata[labelidx] < C);
    if (labeldata[labelidx] == ignore_value) {
      continue;
    }
    if (labeldata[labelidx] > 0) {
      dXdata[Xidx] = -dYdata[i] / max(Xdata[Xidx], log_threshold) / pos;
    } else {
      dXdata[Xidx] = -dYdata[i] / max(Xdata[Xidx], log_threshold) / neg;
    }
  }
}

__global__ void LabelCrossEntropyWSLGradientKernel_CLASSWISE(
    const int outer_size, const int C, const int inner_size, const float* Xdata,
    const int* labeldata, const float* dYdata, const float* countdata,
    const int batch_size, const int num_classes, const float log_threshold,
    const float ignore_value, float* dXdata) {
  int i = blockIdx.x;
  int b = i / num_classes;
  int c = i % num_classes;
  float pos = countdata[b * 2];
  float neg = countdata[b * 2 + 1];
  for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
    int Xidx = i * inner_size + j;
    int labelidx = b * inner_size + j;
    CUDA_KERNEL_ASSERT(labeldata[labelidx] >= 0 && labeldata[labelidx] < C);
    if (labeldata[labelidx] != c) {
      continue;
    }
    if (labeldata[labelidx] == ignore_value) {
      continue;
    }
    if (labeldata[labelidx] > 0) {
      dXdata[Xidx] = -dYdata[i] / max(Xdata[Xidx], log_threshold) / pos;
    } else {
      dXdata[Xidx] = -dYdata[i] / max(Xdata[Xidx], log_threshold) / neg;
    }
  }
}
}  // namespace

template <>
bool LabelCrossEntropyWSLOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), 4);
  CAFFE_ENFORCE_EQ(label.dim(), 4);
  CAFFE_ENFORCE_EQ(X.dim32(0), label.dim32(0));
  CAFFE_ENFORCE_EQ(1, label.dim32(1));
  CAFFE_ENFORCE_EQ(X.dim32(2), label.dim32(2));
  CAFFE_ENFORCE_EQ(X.dim32(3), label.dim32(3));

  const int batch_size = X.dim32(0);
  const int num_classes = X.dim32(1);
  const auto inner_size = X.dim32(2) * X.dim32(3);
  // const auto outer_size = X.dim32(0);
  const auto outer_size = X.numel() / inner_size;

  auto* Y = Output(0);
  auto* count = Output(1);
  if (X.dim() == 0) {
    Y->Resize(std::vector<int64_t>{});
    count->Resize(std::vector<int64_t>{});
  } else {
    std::vector<int64_t> dims(X.sizes().begin(), X.sizes().end() - 2);
    Y->Resize(dims);
    dims.push_back(2);
    count->Resize(dims);
  }
  // Y->Resize(vector<int64_t>(outer_size));
  // count->Resize(vector<int64_t>(outer_size, 2));

  LabelBalanceWSLKernel<<<batch_size, CAFFE_CUDA_NUM_THREADS, 0,
                          context_.cuda_stream()>>>(
      batch_size, inner_size, label.data<int>(), ignore_value_,
      count->mutable_data<float>());

  LabelCrossEntropyWSLKernel_CLASSWISE<<<outer_size, CAFFE_CUDA_NUM_THREADS, 0,
                                         context_.cuda_stream()>>>(
      outer_size, num_classes, inner_size, X.data<float>(), label.data<int>(),
      count->data<float>(), batch_size, num_classes, kLOG_THRESHOLD(),
      ignore_value_, Y->mutable_data<float>());
  return true;
}

template <>
bool LabelCrossEntropyWSLGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto& dY = Input(2);
  auto& count = Input(3);

  const int batch_size = X.dim32(0);
  const int num_classes = X.dim32(1);
  const auto inner_size = X.dim32(2) * X.dim32(3);
  // const auto outer_size = X.dim32(0);
  const auto outer_size = X.numel() / inner_size;

  auto* dX = Output(0);
  dX->ResizeLike(X);
  math::Set<float, CUDAContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                                &context_);
  LabelCrossEntropyWSLGradientKernel_CLASSWISE<<<
      outer_size, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      outer_size, num_classes, inner_size, X.data<float>(), label.data<int>(),
      dY.data<float>(), count.data<float>(), batch_size, num_classes,
      kLOG_THRESHOLD(), ignore_value_, dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(LabelCrossEntropyWSL,
                       LabelCrossEntropyWSLOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(LabelCrossEntropyWSLGradient,
                       LabelCrossEntropyWSLGradientOp<float, CUDAContext>);

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

__global__ void CrossEntropyWithLogitsKernel(const int nthreads,
                                             const float* Xdata,
                                             const float* Ldata,
                                             const float log_threshold,
                                             float* Ydata) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    float prob = max(Xdata[i], log_threshold);
    float one_prob = max(1 - Xdata[i], log_threshold);

    float* address = Ydata + 0;
    float val = -1.0 * (Ldata[i] * log(prob) + (1 - Ldata[i]) * log(one_prob));
    gpu_atomic_add(val, address);
  }
}

__global__ void CrossEntropyWithLogitsGradientKernel(
    const int nthreads, const float* Xdata, const float* Ldata,
    const float* dYdata, const float log_threshold, const float diff_threshold,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    float grad = dYdata[0];
    float prob = max(Xdata[i], log_threshold);
    float one_prob = max(1 - Xdata[i], log_threshold);
    dXdata[i] =
        min(grad * (-1 * Ldata[i] / prob - (-1) * (1 - Ldata[i]) / one_prob),
            diff_threshold);
  }
}

}  // namespace

template <>
bool CrossEntropyWithLogitsOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& L = Input(1);

  // if (InputSize() > 2) {
  // printf("Found unused input in CrossEntropyWithLogits %d\n",
  // InputSize() - 2);
  //}

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(X.sizes(), L.sizes());

  int N = X.dim32(0);

  auto* Y = Output(0);
  Y->Resize(vector<int64_t>{});
  math::Set<float, CUDAContext>(Y->numel(), 0.f, Y->mutable_data<float>(),
                                &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  auto* Ydata = Y->mutable_data<float>();

  CrossEntropyWithLogitsKernel<<<CAFFE_GET_BLOCKS(X.numel()),
                                 CAFFE_CUDA_NUM_THREADS, 0,
                                 context_.cuda_stream()>>>(
      X.numel(), Xdata, Ldata, kLOG_THRESHOLD(), Ydata);

  math::Scale<float, float, CUDAContext>(Y->numel(), float(1.0 / N), Ydata, Ydata,
                                         &context_);

  return true;
}

template <>
bool CrossEntropyWithLogitsGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& L = Input(1);
  auto& dY = Input(2);

  CAFFE_ENFORCE_EQ(X.numel(), L.numel());
  CAFFE_ENFORCE_EQ(X.dim32(0), L.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), L.dim32(1));
  CAFFE_ENFORCE_EQ(dY.numel(), 1);

  int N = X.dim32(0);

  auto* dX = Output(0);
  dX->ResizeLike(X);
  math::Set<float, CUDAContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                                &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  CrossEntropyWithLogitsGradientKernel<<<CAFFE_GET_BLOCKS(X.numel()),
                                         CAFFE_CUDA_NUM_THREADS, 0,
                                         context_.cuda_stream()>>>(
      X.numel(), Xdata, Ldata, dYdata, kLOG_THRESHOLD(), kDIFF_THRESHOLD(),
      dXdata);

  math::Scale<float, float, CUDAContext>(dX->numel(), float(1.0 / N), dXdata, dXdata,
                                         &context_);

  return true;
}

// REGISTER_CUDA_OPERATOR(CrossEntropyWithLogits,
// CrossEntropyWithLogitsOp<float, CUDAContext>);
// REGISTER_CUDA_OPERATOR(CrossEntropyWithLogitsGradient,
// CrossEntropyWithLogitsGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(CrossEntropyWithLogits, GPUFallbackOp);
REGISTER_CUDA_OPERATOR(CrossEntropyWithLogitsGradient, GPUFallbackOp);

REGISTER_CUDA_OPERATOR(WeightedCrossEntropyWithLogits, GPUFallbackOp);
REGISTER_CUDA_OPERATOR(WeightedCrossEntropyWithLogitsGradient, GPUFallbackOp);

namespace {

__device__ float sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * (tgt - (lgt >= 0)) - log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

__device__ float sigmoid_xent_backward(float lgt, float tgt) {
  return tgt - 1. / (1. + exp(-lgt));
}

__global__ void SigmoidBalanceWSLKernel(const int outer_size,
                                        const int inner_size,
                                        const float* targets_ptr,
                                        const float ignore_value,
                                        float* count_ptr) {
  int i = blockIdx.x;
  int last_idx = (i + 1) * inner_size;
  float pos = 0;
  float neg = 0;
  for (int in_idx = i * inner_size + threadIdx.x; in_idx < last_idx;
       in_idx += blockDim.x) {
    if (targets_ptr[in_idx] == ignore_value) {
      continue;
    }
    if (targets_ptr[in_idx] > 0.5) {
      pos += 1;
    } else {
      neg += 1;
    }
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float pos_sum = BlockReduce(temp_storage).Sum(pos);

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce2;
  __shared__ typename BlockReduce2::TempStorage temp_storage2;
  float neg_sum = BlockReduce2(temp_storage2).Sum(neg);

  if (threadIdx.x == 0) {
    count_ptr[i * 2] = pos_sum;
    count_ptr[i * 2 + 1] = neg_sum;
  }
}

__global__ void SigmoidCrossEntropyWithLogitsWSLKernel(
    const int outer_size, const int inner_size, const float* logits_ptr,
    const float* targets_ptr, const float* count_ptr, const float ignore_value,
    float* out_ptr) {
  int i = blockIdx.x;
  int last_idx = (i + 1) * inner_size;
  float value = 0;
  float pos = count_ptr[i * 2];
  float neg = count_ptr[i * 2 + 1];
  for (int in_idx = i * inner_size + threadIdx.x; in_idx < last_idx;
       in_idx += blockDim.x) {
    if (targets_ptr[in_idx] == ignore_value) {
      continue;
    }
    if (targets_ptr[in_idx] > 0.5) {
      value +=
          sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]) / pos;
    } else {
      value +=
          sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]) / neg;
    }
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float sum = BlockReduce(temp_storage).Sum(value);
  if (threadIdx.x == 0) {
    out_ptr[i] = -sum;
  }
}

__global__ void SigmoidCrossEntropyWithLogitsWSLGradientKernel(
    const int outer_size, const int inner_size, const float* g_ptr,
    const float* logits_ptr, const float* targets_ptr, const float* count_ptr,
    const float ignore_value, float* out_ptr) {
  CUDA_1D_KERNEL_LOOP(in_idx, outer_size * inner_size) {
    int i = in_idx / inner_size;
    if (targets_ptr[in_idx] == ignore_value) {
      out_ptr[in_idx] = 0.0;
      continue;
    }
    // auto g_factor = -g_ptr[i] / inner_size;
    float g_factor;
    float count;
    if (targets_ptr[in_idx] > 0.5) {
      count = count_ptr[i * 2];
    } else {
      count = count_ptr[i * 2 + 1];
    }
    if (count > 0) {
      g_factor = -g_ptr[i] / count;
    } else {
      g_factor = 0;
    }
    out_ptr[in_idx] = g_factor * sigmoid_xent_backward(logits_ptr[in_idx],
                                                       targets_ptr[in_idx]);
  }
}
}  // namespace

template <>
bool SigmoidCrossEntropyWithLogitsWSLOp<float, CUDAContext>::RunOnDevice() {
  auto& logits = Input(0);
  auto& targets = Input(1);
  CAFFE_ENFORCE(logits.sizes() == targets.sizes());
  // const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto inner_size = logits.dim32(2) * logits.dim32(3);
  const auto outer_size = logits.numel() / inner_size;

  auto* out = Output(0);
  auto* count = Output(1);
  if (logits.dim() == 0) {
    out->Resize(std::vector<int64_t>{});
    count->Resize(std::vector<int64_t>{});
  } else {
    std::vector<int64_t> dims(logits.sizes().begin(), logits.sizes().end() - 2);
    out->Resize(dims);
    dims.push_back(2);
    count->Resize(dims);
  }
  auto* out_ptr = out->mutable_data<float>();
  auto* count_ptr = count->mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();

  if (logits.numel() <= 0) {
    // nothing to do, not even launching kernel
    return true;
  }

  SigmoidBalanceWSLKernel<<<outer_size, CAFFE_CUDA_NUM_THREADS, 0,
                            context_.cuda_stream()>>>(
      outer_size, inner_size, targets_ptr, ignore_value_, count_ptr);

  SigmoidCrossEntropyWithLogitsWSLKernel<<<outer_size, CAFFE_CUDA_NUM_THREADS,
                                           0, context_.cuda_stream()>>>(
      outer_size, inner_size, logits_ptr, targets_ptr, count_ptr, ignore_value_,
      out_ptr);
  return true;
}

template <>
bool SigmoidCrossEntropyWithLogitsWSLGradientOp<float,
                                                CUDAContext>::RunOnDevice() {
  auto& g = Input(0);
  auto& logits = Input(1);
  auto& targets = Input(2);
  auto& count = Input(3);
  CAFFE_ENFORCE(logits.sizes() == targets.sizes());
  // const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto inner_size = logits.dim32(2) * logits.dim32(3);
  const auto outer_size = logits.numel() / inner_size;
  CAFFE_ENFORCE(g.numel() == outer_size);

  auto* out = Output(0);
  out->ResizeLike(logits);
  auto* out_ptr = out->mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* g_ptr = g.data<float>();
  auto* count_ptr = count.data<float>();

  SigmoidCrossEntropyWithLogitsWSLGradientKernel<<<
      CAFFE_GET_BLOCKS(outer_size * inner_size), CAFFE_CUDA_NUM_THREADS, 0,
      context_.cuda_stream()>>>(outer_size, inner_size, g_ptr, logits_ptr,
                                targets_ptr, count_ptr, ignore_value_, out_ptr);
  return true;
}

REGISTER_CUDA_OPERATOR(SigmoidCrossEntropyWithLogitsWSL,
                       SigmoidCrossEntropyWithLogitsWSLOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SigmoidCrossEntropyWithLogitsWSLGradient,
    SigmoidCrossEntropyWithLogitsWSLGradientOp<float, CUDAContext>);

}  // namespace caffe2
