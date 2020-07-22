#include <functional>

#include "caffe2/core/context_gpu.h"
#include "cpg_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void kernel_show(const T* Xdata, const int batch_size,
                            const int channels, const int height,
                            const int width, const int ndim, const int gpu_id,
                            const int uuid) {
  printf("uuid=%d gpu=%d ndim=%d b = %d c = %d h = %d w = %d\n", uuid, gpu_id,
         ndim, batch_size, channels, height, width);
  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index_X = ((b * channels + c) * height + h) * width + w;
          printf("b = %d c = %d h = %d w = %d %.32f\n", b, c, h, w,
                 Xdata[index_X]);
        }
      }
    }
  }
}

template <typename T>
__global__ void set_grad(const int nthreads, const T* Pdata,
                         const int num_classes, const int label_idx, T* Gdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % num_classes;

    if (c == label_idx) {
      Gdata[index] = Pdata[index];
    } else {
      Gdata[index] = 0;
    }
  }
}

// Implemented in pytorch v1.1.0
// const Tensor& BlobGetTensor(const Blob& blob, DeviceType device_type) {
//  if (blob.IsType<Tensor>()) {
//    const auto& tensor = blob.Get<Tensor>();
//    if (tensor.GetDeviceType() == device_type) {
//      return tensor;
//    }
//  }
//  CAFFE_THROW("Blob didn't contain a Tensor or the device_type doesn't
//  match");
//}

}  // namespace

template <>
bool CPGOp<float, CUDAContext>::RunOnDevice() {
  const int gpu_id = context_.device_id();
  const string namescope = "gpu_" + to_string(gpu_id);

  const string pred_blob_name = namescope + "/" + pred_blob_name_;
  const string pred_grad_blob_name =
      namescope + "/" + pred_blob_name_ + "_grad";
  const string data_blob_name = namescope + "/" + data_blob_name_;
  const string data_grad_blob_name =
      namescope + "/" + data_blob_name_ + "_grad";
  const string cpg_net_name = namescope + "_" + cpg_net_name_;

  const auto& X = Input(0);
  const auto& Y = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(Y.dim(), 2);
  CAFFE_ENFORCE_EQ(X.dim32(0), Y.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), Y.dim32(1));

  const int batch_size = X.dim32(0);
  const int num_classes = X.dim32(1);

  if (cur_iter_ >= max_iter_) {
    const Blob* data_blob = gWorkspace_->GetBlob(data_blob_name);
    // const auto& D = data_blob->Get<Tensor>();
    const Tensor& D = BlobGetTensor(*data_blob, caffe2::CUDA);
    const int data_height = D.dim32(2);
    const int data_width = D.dim32(3);
    auto* M = Output(
        0, vector<int64_t>{batch_size, num_classes, data_height, data_width},
        at::dtype<float>());
    // M->Resize(batch_size, num_classes, data_height, data_width);
    math::Set<float, CUDAContext>(M->numel(), 0.f, M->mutable_data<float>(),
                                  &context_);
    // M->Resize(batch_size, num_classes, 1, 1);
    // gWorkspace_->DeleteNet(cpg_net_name);
    return true;
  }

  const Blob* pred_blob = gWorkspace_->GetBlob(pred_blob_name);
  // const caffe2::TensorCUDA& P = pred_blob->Get<caffe2::TensorCUDA>();
  const Tensor& P = BlobGetTensor(*pred_blob, caffe2::CUDA);
  const float* Pdata = P.data<float>();

  Blob* pred_grad_blob = gWorkspace_->GetBlob(pred_grad_blob_name);
  // caffe2::TensorCUDA* G = pred_grad_blob->GetMutable<caffe2::TensorCUDA>();
  Tensor* G = BlobGetMutableTensor(pred_grad_blob, P.sizes(),
                                   at::dtype(P.dtype()).device(caffe2::CUDA));
  // G->ResizeLike(P);
  float* Gdata = G->mutable_data<float>();

  const Blob* data_blob = gWorkspace_->GetBlob(data_blob_name);
  // const auto& D = data_blob->Get<Tensor>();
  const Tensor& D = BlobGetTensor(*data_blob, caffe2::CUDA);
  CAFFE_ENFORCE_EQ(D.dim32(0), batch_size);
  CAFFE_ENFORCE_EQ(D.dim32(1), 3);
  const int data_height = D.dim32(2);
  const int data_width = D.dim32(3);
  auto* M = Output(
      0, vector<int64_t>{batch_size, num_classes, data_height, data_width},
      at::dtype<float>());
  // M->Resize(batch_size, num_classes, data_height, data_width);
  math::Set<float, CUDAContext>(M->numel(), 0.f, M->mutable_data<float>(),
                                &context_);

  NetBase* cpg_net = gWorkspace_->GetNet(cpg_net_name);
  if (cpg_net == nullptr) {
    std::cout << "No Found Net: " << cpg_net_name << std::endl;
    return false;
  }

  int uuid;
  if (debug_info_) {
    srand(time(NULL));
    uuid = rand();
    kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0, context_.cuda_stream()>>>(
        X.data<float>(), batch_size, num_classes, 1, 1, X.dim(), gpu_id, uuid);
    kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0, context_.cuda_stream()>>>(
        Y.data<float>(), batch_size, num_classes, 1, 1, Y.dim(), gpu_id, uuid);
  }

  Tensor Xcpu = Tensor(X, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Xcpudata = Xcpu.data<float>();

  Tensor Ycpu = Tensor(Y, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Ycpudata = Ycpu.data<float>();

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < num_classes; c++) {
      int label_idx = b * num_classes + c;
      float label_value = Xcpudata[label_idx];
      float pred_value = Ycpudata[label_idx];
      if (debug_info_) {
        printf("uuid %d gpu %d b %d c %d: %.32f %.32f\n", uuid, gpu_id, b, c,
               label_value, pred_value);
      }
      if (label_value < 0.5) {
        continue;
      }
      if (pred_value < tau_) {
        continue;
      }
      if (pred_value >= 0.99999) {
        pred_value = 0.99999;
      }
      // math::Set<float, CUDAContext>(G->numel(), 0.f, Gdata, &context_);
      // math::Set<float, CUDAContext>(1, pred_value, Gdata + label_idx,
      //&context_);

      set_grad<float><<<CAFFE_GET_BLOCKS(G->numel()), CAFFE_CUDA_NUM_THREADS, 0,
                        context_.cuda_stream()>>>(
          G->numel(), Pdata, num_classes, label_idx, Gdata);

      if (debug_info_) {
        std::cout << "Run Net: " << cpg_net_name << std::endl;
      }
      if (!cpg_net->Run()) {
        // if (!gWorkspace_->RunNet(cpg_net_name)) {
        std::cout << "Error in Running Net: " << cpg_net_name << std::endl;
        return false;
      }
      cpg_net->Wait();
      context_.FinishDeviceComputation();

      if (debug_info_) {
        std::cout << "Get Data: " << data_grad_blob_name << std::endl;
      }
      Blob* data_grad_blob = gWorkspace_->GetBlob(data_grad_blob_name);
      auto* dD = data_grad_blob->GetMutable<Tensor>();
      if (data_grad_blob->IsType<Tensor>() && *dD &&
          dD->GetDeviceType() == caffe2::CUDA) {
      } else {
        CAFFE_THROW(
            "Blob didn't contain a Tensor or the device_type doesn't match");
      }

      math::Abs<float, CUDAContext>(dD->numel(), dD->data<float>(),
                                    dD->mutable_data<float>(), &context_);

      const float* dDdata = dD->data<float>();
      float* Mdata =
          M->mutable_data<float>() + data_height * data_width * label_idx;

      // if it has 3 channel
      math::Max<float, CUDAContext>(
          data_height * data_width, dDdata + data_height * data_width * 0,
          dDdata + data_height * data_width * 1, Mdata, &context_);
      math::Max<float, CUDAContext>(data_height * data_width,
                                    dDdata + data_height * data_width * 2,
                                    Mdata, Mdata, &context_);
    }
  }

  cur_iter_++;

  return true;
}

REGISTER_CUDA_OPERATOR(CPG, CPGOp<float, CUDAContext>);

}  // namespace caffe2
