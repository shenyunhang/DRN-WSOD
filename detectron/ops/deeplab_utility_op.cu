#include <cfloat>
#include <functional>

#include "caffe2/core/context_gpu.h"
#include "deeplab_utility_op.h"

namespace caffe2 {

namespace {

__global__ void Softmax_Kernel(const int nthreads, const float* CPGdata,
                               const float* Ldata, const float* Pdata,
                               const int batch_size, const int num_classes,
                               const int height, const int width,
                               const float tau, const float fg_th,
                               const float bg_th, int* MLdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int b = index / height / width;

    int class_idx = 0;
    int fg_num = 0;
    int bg_num = 0;
    int ig_num = 0;
    for (int c = 0; c < num_classes; c++) {
      if (Ldata[b * num_classes + c] == 0) {
        continue;
      }

      if (CPGdata[index] >= fg_th) {
        if (Pdata[b * num_classes + c] < tau) {
          ig_num++;
        } else if (Ldata[b * num_classes + c] == 0.5) {
          ig_num++;
        } else {
          fg_num++;
          class_idx = c + 1;
        }
      } else if (CPGdata[index] <= bg_th) {
        bg_num++;
      } else {
        ig_num++;
        break;
      }
    }

    if (ig_num == 0) {
      if (fg_num == 0) {
        MLdata[index] = 0;
      } else if (fg_num == 1) {
        MLdata[index] = class_idx;
      } else if (fg_num > 1) {
        MLdata[index] = -1;
      }
    } else {
      MLdata[index] = -1;
    }
  }
}

__global__ void Sigmoid_Kernel(const int nthreads, const float* CPGdata,
                               const float* Ldata, const float* Pdata,
                               const int batch_size, const int num_classes,
                               const int height, const int width,
                               const float tau, const float fg_th,
                               const float bg_th, float* MLdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = (index / height / width) % num_classes;
    int b = index / height / width / num_classes;

    if (Ldata[b * num_classes + c] == 0) {
      MLdata[index] = 0.0;
      continue;
    }
    if (Ldata[b * num_classes + c] == 0.5) {
      MLdata[index] = 0.5;
      continue;
    }
    if (Pdata[b * num_classes + c] < tau) {
      MLdata[index] = 0.5;
      continue;
    }

    if (CPGdata[index] >= fg_th) {
      MLdata[index] = 1.0;
      continue;
    } else if (CPGdata[index] <= bg_th) {
      MLdata[index] = 0.0;
      continue;
    } else {
      MLdata[index] = 0.5;
      continue;
    }
  }
}

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
bool DeeplabUtilityOp<float, CUDAContext>::RunOnDevice() {
  const auto& CPG = Input(0);
  const auto& L = Input(1);
  const auto& P = Input(2);

  CAFFE_ENFORCE_EQ(CPG.dim(), 4);
  CAFFE_ENFORCE_EQ(L.dim(), 2);
  CAFFE_ENFORCE_EQ(CPG.dim32(0), L.dim32(0));
  CAFFE_ENFORCE_EQ(CPG.dim32(1), L.dim32(1));

  const int batch_size = CPG.dim32(0);
  const int num_classes = CPG.dim32(1);
  const int height = CPG.dim32(2);
  const int width = CPG.dim32(3);

  const float* CPGdata = CPG.data<float>();
  const float* Ldata = L.data<float>();
  const float* Pdata = P.data<float>();

  auto* ML = Output(0);
  if (softmax_) {
    ML->Resize(batch_size, 1, height, width);
  } else {
    ML->ResizeLike(CPG);
  }

  // Get max value
  // TensorCPU CPGcpu = Tensor<CPUContext>(CPG, &context_);
  // context_.FinishDeviceComputation();
  // float max_val = get_max<float>(CPGcpu.numel(), CPGcpu.data<float>());
  float max_val = 1.;

  if (softmax_) {
    Softmax_Kernel<<<CAFFE_GET_BLOCKS(ML->numel()), CAFFE_CUDA_NUM_THREADS, 0,
                     context_.cuda_stream()>>>(
        ML->numel(), CPGdata, Ldata, Pdata, batch_size, num_classes, height,
        width, tau_, fg_th_ * max_val, bg_th_ * max_val,
        ML->mutable_data<int>());
  } else {
    Sigmoid_Kernel<<<CAFFE_GET_BLOCKS(ML->numel()), CAFFE_CUDA_NUM_THREADS, 0,
                     context_.cuda_stream()>>>(
        ML->numel(), CPGdata, Ldata, Pdata, batch_size, num_classes, height,
        width, tau_, fg_th_ * max_val, bg_th_ * max_val,
        ML->mutable_data<float>());
  }

  return true;
}

REGISTER_CUDA_OPERATOR(DeeplabUtility, DeeplabUtilityOp<float, CUDAContext>);

}  // namespace caffe2
