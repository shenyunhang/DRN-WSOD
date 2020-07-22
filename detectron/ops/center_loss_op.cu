#include <cfloat>
#include <functional>

#include "caffe2/core/context_gpu.h"
#include "center_loss_op.h"

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

}  // namespace

template <>
bool CenterLossOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& P = Input(1);
  const auto& F = Input(2);
  const auto& CF = Input(3);
  const auto& dCF = Input(4);
  const auto& ndCF = Input(5);
  CAFFE_ENFORCE_EQ(X.dim32(1), P.dim32(1));
  CAFFE_ENFORCE_EQ(X.dim32(1), CF.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), dCF.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), ndCF.dim32(0));
  CAFFE_ENFORCE_EQ(P.dim32(0), F.dim32(0));
  CAFFE_ENFORCE_EQ(F.dim32(1), CF.dim32(2));
  CAFFE_ENFORCE_EQ(F.dim32(1), dCF.dim32(2));
  CAFFE_ENFORCE_EQ(CF.dim32(1), dCF.dim32(1));
  CAFFE_ENFORCE_EQ(CF.dim32(1), ndCF.dim32(1));
  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(P.dim(), 2);
  CAFFE_ENFORCE_EQ(F.dim(), 2);
  CAFFE_ENFORCE_EQ(CF.dim(), 3);
  CAFFE_ENFORCE_EQ(dCF.dim(), 3);
  CAFFE_ENFORCE_EQ(ndCF.dim(), 2);

  const int batch_size = X.dim32(0);
  const int num_classes = X.dim32(1);
  const int num_rois = P.dim32(0);
  const int num_center = CF.dim32(1);
  const int feature_dim = CF.dim32(2);

  auto* L = Output(0);
  auto* D = Output(1);
  auto* S = Output(2);
  L->Resize(1);
  math::Set<float, CUDAContext>(L->numel(), 0.f, L->mutable_data<float>(),
                                &context_);
  D->Resize(num_classes, top_k_, feature_dim);
  math::Set<float, CUDAContext>(D->numel(), 0.f, D->mutable_data<float>(),
                                &context_);
  S->Resize(num_classes);
  math::Set<float, CUDAContext>(S->numel(), -1.f, S->mutable_data<float>(),
                                &context_);

  if (cur_iter_ >= max_iter_) {
    return true;
  }

  const int gpu_id = context_.device_id();
  int uuid;
  if (debug_info_) {
    srand(time(NULL));
    uuid = rand();
  }

  if (init_) {
    for (int c = 0; c < num_classes; c++) {
      set<int> r_s;
      roi_sets_.push_back(r_s);

      vector<int> a_u_c;
      accum_update_class_.push_back(a_u_c);
      for (int m = 0; m < num_center; ++m) {
        accum_update_class_[c].push_back(0);
      }
    }
    accum_loss_ = 0;
    init_ = false;
  }

  Tensor Xcpu = Tensor(X, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Xcpudata = Xcpu.data<float>();

  Tensor Pcpu = Tensor(P, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Pcpudata = Pcpu.data<float>();

  Tensor Dt = Tensor(caffe2::CUDA);
  Dt.Resize(top_k_, feature_dim);

  Tensor statistics_gpu(caffe2::CUDA);
  statistics_gpu.Resize(std::vector<int>{});

  Tensor statistics_cpu(caffe2::CPU);
  statistics_cpu.Resize(std::vector<int>{});

  // get the top_k_ score RoI index
  int num_gt_class = 0;
  for (int b = 0; b < batch_size; b++) {
    CAFFE_ENFORCE_EQ(batch_size, 1);
    for (int c = 0; c < num_classes; c++) {
      roi_sets_[c].clear();

      if (c == ignore_label_) {
        continue;
      }

      if (num_rois < top_k_) {
        continue;
      }

      int label_idx = b * num_classes + c;
      float label_val = Xcpudata[label_idx];

      if (debug_info_) {
        printf("uuid %d gpu %d b %d c %d label_val %.32f\n", uuid, gpu_id, b, c,
               label_val);
      }

      if (label_val < 0.5) {
        continue;
      }

      num_gt_class++;
      for (int k = 0; k < top_k_; k++) {
        float max_val = -FLT_MAX;
        int max_idx = -1;
        for (int r = 0; r < num_rois; ++r) {
          int pred_idx = r * num_classes + c;
          float pred_val = Pcpudata[pred_idx];
          if (max_val < pred_val) {
            if (roi_sets_[c].find(r) == roi_sets_[c].end()) {
              max_val = pred_val;
              max_idx = r;
            }
          }
        }

        if (max_idx == -1) {
          printf("can not find enought roi.\n");
          // roi_sets_[c].clear();
          // break;
          return false;
        }
        roi_sets_[c].insert(max_idx);
        if (debug_info_) {
          printf("uuid %d gpu %d b %d c %d max_val %.32f max_idx %d\n", uuid,
                 gpu_id, b, c, max_val, max_idx);
        }
      }
    }
  }

  const float* Fdata = F.data<float>();
  const float* CFdata = CF.data<float>();
  float* Ddata = D->mutable_data<float>();
  float* Dtdata = Dt.mutable_data<float>();

  // TODO(YH): BUG, RoI score is forgot to use.
  float dot = 0;
  for (int b = 0; b < batch_size; b++) {
    CAFFE_ENFORCE_EQ(batch_size, 1);
    for (int c = 0; c < num_classes; c++) {
      int center_selector = -1;
      if (roi_sets_[c].size() == 0) continue;
      float c_dot = FLT_MAX;
      for (int m = 0; m < num_center; ++m) {
        set<int>::iterator it;
        int k;
        for (k = 0, it = roi_sets_[c].begin(); it != roi_sets_[c].end();
             ++k, ++it) {
          int r = *it;
          math::Sub<float, CUDAContext>(
              feature_dim, Fdata + r * feature_dim,
              CFdata + (c * num_center + m) * feature_dim,
              Dtdata + k * feature_dim, &context_);
        }

        math::Dot<float, CUDAContext>(top_k_ * feature_dim, Dtdata, Dtdata,
                                      statistics_gpu.mutable_data<float>(),
                                      &context_);
        statistics_cpu.CopyFrom(statistics_gpu, false);
        context_.FinishDeviceComputation();
        const float* statistics_cpu_data = statistics_cpu.data<float>();
        float cm_dot = statistics_cpu_data[0];

        if (cm_dot < c_dot) {
          math::Axpby<float, float, CUDAContext>(
              Dt.numel(), float(1), Dtdata, float(0),
              Ddata + (c * top_k_ + 0) * feature_dim, &context_);
          c_dot = cm_dot;

          center_selector = m;
          math::Set<float, CUDAContext>(
              1, (float)(m), S->mutable_data<float>() + c, &context_);
        }
      }
      accum_update_class_[c][center_selector]++;
      dot += c_dot;
    }
  }

  float loss = num_gt_class > 0
                   ? dot / num_gt_class / top_k_ / feature_dim / (float)(2)
                   : 0;

  if (debug_info_) {
    printf("uuid %d gpu %d loss %.32f num_gt_class %d\n", uuid, gpu_id, loss,
           num_gt_class);
  }

  math::Set<float, CUDAContext>(L->numel(), loss, L->mutable_data<float>(),
                                &context_);

  accum_loss_ += loss;
  cur_iter_++;

  if (cur_iter_ % display_ == 0 || cur_iter_ == 1) {
    std::cout << "CenterLoss #iter_: " << cur_iter_
              << " #loss_: " << accum_loss_
              << " AVE loss: " << accum_loss_ / display_ << std::endl;
    accum_loss_ = 0;

    for (int c = 0; c < num_classes; ++c) {
      for (int m = 0; m < num_center; ++m) {
        std::cout << accum_update_class_[c][m] << "\t";
        accum_update_class_[c][m] = 0;
      }
      std::cout << "\t";
      for (int m = 0; m < num_center; ++m) {
        math::Abs<float, CUDAContext>(
            feature_dim, CFdata + (c * num_center + m) * feature_dim, Dtdata,
            &context_);
        math::Sum<float, CUDAContext>(feature_dim, Dtdata,
                                      statistics_gpu.mutable_data<float>(),
                                      &context_);
        statistics_cpu.CopyFrom(statistics_gpu, false);
        context_.FinishDeviceComputation();
        const float* statistics_cpu_data = statistics_cpu.data<float>();
        float asum = statistics_cpu_data[0];
        std::cout << asum << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  if (debug_info_) {
    printf("Show S\n");
    kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0, context_.cuda_stream()>>>(
        S->data<float>(), num_classes, num_center, 1, 1, S->dim(), gpu_id,
        uuid);
  }

  return true;
}

template <>
bool CenterLossGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& P = Input(1);
  const auto& CF = Input(2);
  const auto& dCF = Input(3);
  const auto& ndCF = Input(4);
  const auto& D = Input(5);
  const auto& S = Input(6);
  const auto& dL = Input(7);
  CAFFE_ENFORCE_EQ(X.dim32(1), P.dim32(1));
  CAFFE_ENFORCE_EQ(X.dim32(1), CF.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), dCF.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), ndCF.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), D.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), S.dim32(0));
  CAFFE_ENFORCE_EQ(CF.dim32(1), dCF.dim32(1));
  CAFFE_ENFORCE_EQ(CF.dim32(1), ndCF.dim32(1));
  CAFFE_ENFORCE_EQ(CF.dim32(2), dCF.dim32(2));
  CAFFE_ENFORCE_EQ(CF.dim32(2), D.dim32(2));
  CAFFE_ENFORCE_EQ(D.dim32(1), top_k_);
  CAFFE_ENFORCE_EQ(dL.dim32(0), 1);
  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(P.dim(), 2);
  CAFFE_ENFORCE_EQ(CF.dim(), 3);
  CAFFE_ENFORCE_EQ(dCF.dim(), 3);
  CAFFE_ENFORCE_EQ(ndCF.dim(), 2);
  CAFFE_ENFORCE_EQ(D.dim(), 3);
  CAFFE_ENFORCE_EQ(S.dim(), 1);
  CAFFE_ENFORCE_EQ(dL.dim(), 1);

  const int batch_size = X.dim32(0);
  const int num_classes = X.dim32(1);
  const int num_rois = P.dim32(0);
  const int num_center = CF.dim32(1);
  const int feature_dim = CF.dim32(2);

  auto* dF = Output(0);
  auto* CFo = Output(1);
  auto* dCFo = Output(2);
  auto* ndCFo = Output(3);
  dF->Resize(num_rois, feature_dim);
  math::Set<float, CUDAContext>(dF->numel(), 0.f, dF->mutable_data<float>(),
                                &context_);

  CAFFE_ENFORCE_EQ(CFo, &CF);
  CAFFE_ENFORCE_EQ(dCFo, &dCF);
  CAFFE_ENFORCE_EQ(ndCFo, &ndCF);

  if (cur_iter_ >= max_iter_) {
    return true;
  }

  const int gpu_id = context_.device_id();
  int uuid;
  if (debug_info_) {
    srand(time(NULL));
    uuid = rand();
  }

  if (init_) {
    for (int c = 0; c < num_classes; c++) {
      set<int> r_s;
      roi_sets_.push_back(r_s);
    }

    dCF_.ResizeLike(dCF);
    ndCF_.ResizeLike(ndCF);

    math::Set<float, CUDAContext>(dCF_.numel(), 0.f, dCF_.mutable_data<float>(),
                                  &context_);
    math::Set<float, CUDAContext>(ndCF_.numel(), 0.f,
                                  ndCF_.mutable_data<float>(), &context_);

    math::Set<float, CUDAContext>(dCFo->numel(), 0.f,
                                  dCFo->mutable_data<float>(), &context_);
    math::Set<float, CUDAContext>(ndCFo->numel(), 0.f,
                                  ndCFo->mutable_data<float>(), &context_);
    init_ = false;
  }

  if (debug_info_) {
    // printf("Show ndCF\n");
    // kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0,
    // context_.cuda_stream()>>>(
    // ndCF.data<float>(), num_classes, num_center, 1, 1, ndCF.dim(), gpu_id,
    // uuid);
    // printf("Show ndCF_\n");
    // kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0,
    // context_.cuda_stream()>>>(
    // ndCF_.data<float>(), num_classes, num_center, 1, 1, ndCF_.dim(),
    // gpu_id, uuid);
  }

  math::Add<float, CUDAContext>(dCF.numel(), dCF.data<float>(),
                                dCF_.data<float>(), dCF_.mutable_data<float>(),
                                &context_);
  math::Add<float, CUDAContext>(ndCF.numel(), ndCF.data<float>(),
                                ndCF_.data<float>(),
                                ndCF_.mutable_data<float>(), &context_);

  math::Set<float, CUDAContext>(dCFo->numel(), 0.f, dCFo->mutable_data<float>(),
                                &context_);
  math::Set<float, CUDAContext>(ndCFo->numel(), 0.f,
                                ndCFo->mutable_data<float>(), &context_);

  if (debug_info_) {
    // printf("Show ndCF\n");
    // kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0,
    // context_.cuda_stream()>>>(
    // ndCF.data<float>(), num_classes, num_center, 1, 1, ndCF.dim(), gpu_id,
    // uuid);
    printf("Show ndCF_\n");
    kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0, context_.cuda_stream()>>>(
        ndCF_.data<float>(), num_classes, num_center, 1, 1, ndCF_.dim(),
        gpu_id, uuid);
  }

  Tensor Xcpu = Tensor(X, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Xcpudata = Xcpu.data<float>();

  Tensor Pcpu = Tensor(P, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Pcpudata = Pcpu.data<float>();

  Tensor statistics_cpu(caffe2::CPU);
  statistics_cpu.Resize(std::vector<int>{});

  // get the top_k_ score RoI index
  int num_gt_class = 0;
  for (int b = 0; b < batch_size; b++) {
    CAFFE_ENFORCE_EQ(batch_size, 1);
    for (int c = 0; c < num_classes; c++) {
      roi_sets_[c].clear();

      if (c == ignore_label_) {
        continue;
      }

      if (num_rois < top_k_) {
        continue;
      }

      int label_idx = b * num_classes + c;
      float label_val = Xcpudata[label_idx];

      if (debug_info_) {
        printf("uuid %d gpu %d b %d c %d label_val %.32f\n", uuid, gpu_id, b, c,
               label_val);
      }

      if (label_val < 0.5) {
        continue;
      }

      num_gt_class++;
      for (int k = 0; k < top_k_; k++) {
        float max_val = -FLT_MAX;
        int max_idx = -1;
        for (int r = 0; r < num_rois; ++r) {
          int pred_idx = r * num_classes + c;
          float pred_val = Pcpudata[pred_idx];
          if (max_val < pred_val) {
            if (roi_sets_[c].find(r) == roi_sets_[c].end()) {
              max_val = pred_val;
              max_idx = r;
            }
          }
        }

        if (max_idx == -1) {
          printf("can not find enought roi.\n");
          // roi_sets_[c].clear();
          // break;
          return false;
        }
        roi_sets_[c].insert(max_idx);
        if (debug_info_) {
          printf("uuid %d gpu %d b %d c %d max_val %.32f max_idx %d\n", uuid,
                 gpu_id, b, c, max_val, max_idx);
        }
      }
    }
  }

  // Update ndCFo
  Tensor Scpu = Tensor(S, caffe2::CPU);
  context_.FinishDeviceComputation();
  const float* Scpudata = Scpu.data<float>();

  Tensor ndCFocpu = Tensor(*ndCFo, caffe2::CPU);
  context_.FinishDeviceComputation();
  float* ndCFocpudata = ndCFocpu.mutable_data<float>();

  for (int b = 0; b < batch_size; b++) {
    CAFFE_ENFORCE_EQ(batch_size, 1);
    for (int c = 0; c < num_classes; c++) {
      int center_selector = int(Scpudata[c]);
      if (center_selector != -1) {
        ndCFocpudata[c * num_center + center_selector] += 1;
        CAFFE_ENFORCE_EQ(roi_sets_[c].size(), top_k_);
      } else {
        CAFFE_ENFORCE_EQ(roi_sets_[c].size(), 0);
      }
    }
  }
  ndCFo->CopyFrom(ndCFocpu, false);
  context_.FinishDeviceComputation();
  if (debug_info_) {
    // printf("Show ndCF\n");
    // kernel_show<float><<<CAFFE_GET_BLOCKS(1), 1, 0,
    // context_.cuda_stream()>>>(
    // ndCF.data<float>(), num_classes, num_center, 1, 1, ndCF.dim(), gpu_id,
    // uuid);
  }

  // Update dF dCFo
  statistics_cpu.CopyFrom(dL, false);
  context_.FinishDeviceComputation();
  const float* statistics_cpu_data = statistics_cpu.data<float>();
  float dLcpu = statistics_cpu_data[0];

  const float alpha =
      num_gt_class > 0 ? dLcpu / num_gt_class / top_k_ / feature_dim : 0;

  if (debug_info_) {
    printf("uuid %d gpu %d alpha %.32f dLcpu %.32f num_gt_class %d\n", uuid,
           gpu_id, alpha, dLcpu, num_gt_class);
  }

  const float* Ddata = D.data<float>();

  float* dFdata = dF->mutable_data<float>();
  float* dCFodata = dCFo->mutable_data<float>();

  for (int b = 0; b < batch_size; b++) {
    CAFFE_ENFORCE_EQ(batch_size, 1);
    for (int c = 0; c < num_classes; ++c) {
      set<int>::iterator it;
      int k;
      for (k = 0, it = roi_sets_[c].begin(); it != roi_sets_[c].end();
           ++k, ++it) {
        int r = *it;
        // feature diff
        math::Axpby<float, float, CUDAContext>(
            feature_dim, alpha, Ddata + (c * top_k_ + k) * feature_dim,
            float(1), dFdata + r * feature_dim, &context_);

        // TODO(YH): whether center update is correct
        // center diff
        math::Axpby<float, float, CUDAContext>(
            feature_dim, float(-1), Ddata + (c * top_k_ + k) * feature_dim,
            float(1),
            dCFodata + (c * num_center + int(Scpudata[c])) * feature_dim,
            &context_);
      }
    }
  }

  cur_iter_++;
  // update center
  if (cur_iter_ % update_ == 0) {
    float* CFodata = CFo->mutable_data<float>();
    float* dCFdata = dCF_.mutable_data<float>();
    Tensor ndCFcpu = Tensor(ndCF_, caffe2::CPU);
    context_.FinishDeviceComputation();
    float* ndCFcpudata = ndCFcpu.mutable_data<float>();
    for (int c = 0; c < num_classes; ++c) {
      for (int m = 0; m < num_center; ++m) {
        int num_update_class = ndCFcpudata[c * num_center + m];
        if (debug_info_) {
          printf("uuid %d gpu %d c %d m %d num_update_class %d\n", uuid, gpu_id,
                 c, m, num_update_class);
        }
        math::Axpby<float, float, CUDAContext>(
            feature_dim, lr_ * float(-1) / (num_update_class * top_k_ + 1),
            dCFdata + (c * num_center + m) * feature_dim, float(1),
            CFodata + (c * num_center + m) * feature_dim, &context_);
      }
    }
    math::Set<float, CUDAContext>(dCF_.numel(), 0.f, dCF_.mutable_data<float>(),
                                  &context_);
    math::Set<float, CUDAContext>(ndCF_.numel(), 0.f,
                                  ndCF_.mutable_data<float>(), &context_);
  }

  return true;
}

REGISTER_CUDA_OPERATOR(CenterLoss, CenterLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(CenterLossGradient,
                       CenterLossGradientOp<float, CUDAContext>);

}  // namespace caffe2
