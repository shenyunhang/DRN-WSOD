#include <cfloat>
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "softmax_with_loss_n_op.h"

namespace caffe2 {

namespace {

__global__ void LabelCrossEntropyKernel(
    const int N,
    const int D,
    const float* logPdata,
    const int* labeldata,
    const float* weights,
    float* Ydata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    CUDA_KERNEL_ASSERT(labeldata[i] >= 0 && labeldata[i] < D);
    float weight = weights ? weights[i] : 1.0;
    Ydata[i] = -logPdata[i * D + labeldata[i]] * weight;
  }
}

__global__ void LabelCrossEntropyGradientKernel(
    const int N,
    const int D,
    const float* Pdata,
    const int* labeldata,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int idx = i * D + labeldata[i];
    dXdata[idx] = Pdata[idx] - 1.;
  }
}

__global__ void LabelCrossEntropyGradientKernelWeighted(
    const int N,
    const int D,
    const float* Pdata,
    const int* labeldata,
    float* dXdata,
    const float* weights) {
  CUDA_1D_KERNEL_LOOP(i, N * D) {
    int row = i / D;
    int d = i % D;
    float val = Pdata[i] - 1.0 * (d == labeldata[row]);
    float weight = weights[row];
    dXdata[i] = val * weight;
  }
}

__global__ void ProbCrossEntropyKernel(
    const int N,
    const int D,
    const float* Pdata,
    const float* labeldata,
    const float* weights,
    float* Ydata) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    float weight = weights ? weights[i] : 1.0;
    float sum = 0.0;
    float total_prob = 0.0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
      int idx = i * D + j;
      CUDA_KERNEL_ASSERT(labeldata[idx] >= 0);
      total_prob += labeldata[idx];
      sum += -logf(fmaxf(Pdata[idx], FLT_MIN)) * labeldata[idx] * weight;
    }
    float tot = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    float total_prob_sum = BlockReduce(temp_storage).Sum(total_prob);
    if (threadIdx.x == 0) {
      Ydata[i] = tot;
      // Sanity check
      CUDA_KERNEL_ASSERT(fabsf(1.0 - total_prob_sum) < 1e-5f);
    }
    __syncthreads();
  }
}

__global__ void ProbCrossEntropyGradientKernel(
    const int N,
    const int D,
    const float* Pdata,
    const float* labeldata,
    float* dXdata,
    const float* weights) {
  if (weights == NULL) {
    CUDA_1D_KERNEL_LOOP(idx, N * D) {
      dXdata[idx] = Pdata[idx] - labeldata[idx];
    }
  } else {
    CUDA_1D_KERNEL_LOOP(idx, N * D) {
      dXdata[idx] = (Pdata[idx] - labeldata[idx]) * weights[idx / D];
    }
  }
}

__global__ void SoftmaxNormalizeLogsKernel(
    const int nthreads,
    const int D,
    const float* logits,
    const float* rowmax,
    const float* scales,
    float* out_log) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / D;
    out_log[index] = logits[index] - rowmax[n] - logf(fmaxf(scales[n], FLT_MIN));
  }
}

__global__ void SoftmaxNormalizeKernel(
    const int nthreads,
    const int D,
    const float* probs,
    const float* scales,
    float* out) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / D;
    out[index] = probs[index] / scales[n];
  }
}

void Softmax(
    const int N,
    const int D,
    const float* logits,
    const float* sum_multiplier,
    float* scales,
    float* rowmax,
    float* probs,
    bool log_softmax,
    CUDAContext* context) {
  const int size = N * D;

  math::RowwiseMax<float, CUDAContext>(N, D, logits, rowmax, context);
  // Put the intermediate result X - max(X) into Y
  context->CopySameDevice<float>(size, logits, probs);
  // Subtract the scale
  math::Gemm<float, CUDAContext>(
      CblasNoTrans,
      CblasNoTrans,
      N,
      D,
      1,
      -1,
      rowmax,
      sum_multiplier,
      1,
      probs,
      context);
  // Exponentiation
  math::Exp<float, CUDAContext>(size, probs, probs, context);
  // Sum exponentiated values
  math::Gemv<float, CUDAContext>(CblasNoTrans, N, D, 1, probs, sum_multiplier,
                                 0, scales, context);
  // Normalize
  if (!log_softmax) {
    SoftmaxNormalizeKernel<<<
        CAFFE_GET_BLOCKS(size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(size, D, probs, scales, probs);
  } else {
    SoftmaxNormalizeLogsKernel<<<
        CAFFE_GET_BLOCKS(size),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context->cuda_stream()>>>(size, D, logits, rowmax, scales, probs);
  }
}

__global__ void ValidNormKernel(const int N, const float* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] > 1e-12 ? 1 : 0;
  }
}

} // namespace

template<>
bool SoftmaxWithLossNOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Logits
  auto& T = Input(1);  // Labels / targets

  const float* weights = (InputSize() > 2 ? Input(2).data<float>() : NULL);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);

  auto* P =
      Output(0, X.sizes(), at::dtype<float>()); // Probabilities from softmax
  ReinitializeTensor(&total_weight_ptr_, {1}, at::dtype<float>().device(CUDA));
  total_weight_ptr_.Resize(1);

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.dim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.dim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.numel(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  auto* avg_loss =
      Output(1, vector<int64_t>(), at::dtype<float>()); // Average loss
  if (!losses_.defined()) {
    losses_ = caffe2::empty({N}, at::dtype<float>().device(CUDA));
  } else if (losses_.numel() != N) {
    losses_.Resize(N);
  }

  if (!rowmax_.defined()) {
    rowmax_ = caffe2::empty({N}, at::dtype<float>().device(CUDA));
  } else if (rowmax_.numel() != N) {
    rowmax_.Resize(N);
  }

  if (!sum_multiplier_.defined()) {
    sum_multiplier_ = caffe2::empty({D}, at::dtype<float>().device(CUDA));
    math::Set<float, CUDAContext>(D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  } else if (sum_multiplier_.numel() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CUDAContext>(D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
  }

  Softmax(
      N,
      D,
      X.data<float>(),
      sum_multiplier_.data<float>(),
      losses_.mutable_data<float>(),
      rowmax_.mutable_data<float>(),
      P->template mutable_data<float>(),
      !label_prob_mode_, // logarithmic output
      &context_);
  // Compute label xent loss per example
  if (!label_prob_mode_) {
    LabelCrossEntropyKernel<<<
        CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        D,
        P->data<float>(),
        T.data<int>(),
        weights,
        losses_.mutable_data<float>());
    // Since we had logarithmic output, we need to exponentiate
    // them again.
    math::Exp<float, CUDAContext>(
        N * D, P->data<float>(), P->template mutable_data<float>(), &context_);
  } else {
    ProbCrossEntropyKernel<<<
        std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        D,
        P->data<float>(),
        T.data<float>(),
        weights,
        losses_.mutable_data<float>());
  }

  float total_weight = N;
  if (weights) {

    weights_valid_ = Tensor(caffe2::CUDA);
    weights_valid_.Resize(N);
    ValidNormKernel<<<N, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(N, weights, weights_valid_.mutable_data<float>());

    // Sum weights
    math::Sum<float, CUDAContext>(
        N, weights_valid_.data<float>(), total_weight_ptr_.mutable_data<float>(), &context_, &scratch_);
    CUDA_CHECK(cudaMemcpyAsync(
        &total_weight,
        total_weight_ptr_.data<float>(),
        sizeof(float),
        cudaMemcpyDeviceToHost,
        context_.cuda_stream()));
  }

  // Sum of all losses
  float* avg_loss_data = avg_loss->template mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.numel(), losses_.data<float>(), avg_loss_data, &context_, &scratch_);
  // Average of input batch size
  if (total_weight > 0) {
    math::Scale<float, float, CUDAContext>(
        1, scale_ / total_weight, avg_loss_data, avg_loss_data, &context_);
  }
  if (OutputSize() > 2) {
    OutputTensorAlias(2, losses_);
  }
  return true;
}

template <>
bool SoftmaxWithLossNGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Logits
  auto& T = Input(1);  // Labels / targets
  // Input(2) is weights, if given
  auto& P = Input(InputSize() - 2);  // Probabilities from softmax
  auto& d_avg_loss = Input(InputSize() - 1); // Gradient w.r.t. avg loss
  const float* weights = (InputSize() > 4 ? Input(2).data<float>() : NULL);

  Tensor* dX;
  if (only_loss_) {
    // Memory saving trick to share the buffer with the softmax output.
    // Softmax output is thus overwritten.
    dX = OutputTensorAlias(0, P);
    dX->ResizeLike(X);
  } else {
    dX = Output(0, X.sizes(), at::dtype<float>());
  }

  const auto canonical_axis = X.canonical_axis_index(axis_);
  int N, D;
  N = X.size_to_dim(canonical_axis); // batch size
  D = X.size_from_dim(canonical_axis);

  ReinitializeTensor(&total_weight_ptr_, {1}, at::dtype<float>().device(CUDA));

  if (label_prob_mode_) {
    CAFFE_ENFORCE_GE(T.dim(), 2);
    CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
  } else {
    if (T.dim() == canonical_axis) {
      CAFFE_ENFORCE_EQ(T.numel(), N);
    } else {
      CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
      CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
    }
  }

  // Subtract 1 from labeled positions
  if (!label_prob_mode_) {
    if (weights == nullptr) {
      // Copy softmax probabilities into dX
      if (!only_loss_) {
        context_.CopySameDevice<float>(
            P.numel(), P.data<float>(), dX->template mutable_data<float>());
      }
      LabelCrossEntropyGradientKernel<<<
          CAFFE_GET_BLOCKS(N),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          N,
          D,
          P.data<float>(),
          T.data<int>(),
          dX->template mutable_data<float>());
    } else {
      // Weighted version gets the Pdata values internally
      LabelCrossEntropyGradientKernelWeighted<<<
          CAFFE_GET_BLOCKS(N * D),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          N,
          D,
          P.data<float>(),
          T.data<int>(),
          dX->template mutable_data<float>(),
          weights);
    }
  } else {
    ProbCrossEntropyGradientKernel<<<
        CAFFE_GET_BLOCKS(N * D),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        D,
        P.data<float>(),
        T.data<float>(),
        dX->template mutable_data<float>(),
        weights);
  }
  float total_weight = N;
  if (weights) {

    weights_valid_ = Tensor(caffe2::CUDA);
    weights_valid_.Resize(N);
    ValidNormKernel<<<N, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(N, weights, weights_valid_.mutable_data<float>());

    // Sum weights
    math::Sum<float, CUDAContext>(
        N, weights_valid_.data<float>(), total_weight_ptr_.mutable_data<float>(), &context_, &scratch_);
    CUDA_CHECK(cudaMemcpyAsync(
        &total_weight,
        total_weight_ptr_.data<float>(),
        sizeof(float),
        cudaMemcpyDeviceToHost,
        context_.cuda_stream()));
  }

  // Scale by d_avg_loss / N
  if (total_weight > 0) {
    math::Scale<float, float, CUDAContext>(
        dX->numel(),
        scale_ / total_weight,
        dX->data<float>(),
        dX->template mutable_data<float>(),
        &context_);
  }
  math::Scale<float, float, CUDAContext>(
      dX->numel(),
      d_avg_loss.data<float>(),
      dX->data<float>(),
      dX->template mutable_data<float>(),
      &context_);

  return true;
}

REGISTER_CUDA_OPERATOR(SoftmaxWithLossN,
                       SoftmaxWithLossNOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SoftmaxWithLossNGradient,
                       SoftmaxWithLossNGradientOp<float, CUDAContext>);

} // namespace caffe2
