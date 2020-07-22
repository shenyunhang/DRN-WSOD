#include "pcl_loss_op.h"
#include <math.h>

namespace caffe2 {

namespace {}

template <>
bool PCLLossOp<float, CPUContext>::RunOnDevice() {
  const auto& pcl_probs = Input(0);
  const auto& labels = Input(1);
  const auto& cls_loss_weights = Input(2);
  const auto& gt_assignment = Input(3);
  const auto& pc_labels = Input(4);
  const auto& pc_probs = Input(5);
  const auto& pc_count = Input(6);
  const auto& img_cls_loss_weights = Input(7);
  const auto& im_labels = Input(8);

  // Grab the input tensor
  const float* prob_data_flat = pcl_probs.data<float>();
  const float* labels_flat = labels.data<float>();
  const float* cls_loss_weights_flat = cls_loss_weights.data<float>();
  const float* pc_labels_flat = pc_labels.data<float>();
  const float* pc_probs_flat = pc_probs.data<float>();
  const float* img_cls_loss_weights_flat = img_cls_loss_weights.data<float>();
  const float* im_labels_flat = im_labels.data<float>();

  auto* loss = Output(0);
  loss->Resize(vector<int64_t>());
  math::Set<float, CPUContext>(loss->numel(), 0.f, loss->mutable_data<float>(),
                               &context_);
  float* loss_flat = loss->mutable_data<float>();

  Tensor output = Tensor(caffe2::CPU);
  output.Resize(1, pcl_probs.dim32(1));
  math::Set<float, CPUContext>(output.numel(), 0.f,
                               output.mutable_data<float>(), &context_);

  float* output_flat = output.mutable_data<float>();

  int batch_size = pcl_probs.dim32(0);
  int channels = pcl_probs.dim32(1);
  int num_positive = pc_labels.dim32(1);


    float eps = 1e-6;

    for (int c = 0; c < channels; c++) {
        output_flat[c] = 0;
        if (im_labels_flat[c] != 0) {
            if (c == 0) {
                for (int i = 0; i < batch_size; i++) {
                    if (labels_flat[i] == 0) {
                        output_flat[c] -= cls_loss_weights_flat[i] * log(fmaxf(prob_data_flat[i * channels + c], eps));
                    }
                }
            }
            else {
                for (int i = 0; i < num_positive; i++) {
                    if (pc_labels_flat[i] == c) {
                        output_flat[c] -= img_cls_loss_weights_flat[i] * log(fmaxf(pc_probs_flat[i], eps));
                    }
                }
            }
        }
    }


  math::Sum<float, CPUContext>(output.numel(), output_flat, loss_flat,
                               &context_);
  math::Scale<float, float, CPUContext>(loss->numel(),
                                        float(1.0 / pcl_probs.dim32(0)),
                                        loss_flat, loss_flat, &context_);

  return true;
}

template <>
bool PCLLossGradientOp<float, CPUContext>::RunOnDevice() {
  auto& pcl_probs = Input(0);
  auto& labels = Input(1);
  auto& cls_loss_weights = Input(2);
  auto& gt_assignment = Input(3);
  auto& pc_labels = Input(4);
  auto& pc_probs = Input(5);
  auto& pc_count = Input(6);
  auto& img_cls_loss_weights = Input(7);
  auto& im_labels = Input(8);

  const float* prob_data_flat = pcl_probs.data<float>();
  const float* labels_flat = labels.data<float>();
  const float* cls_loss_weights_flat = cls_loss_weights.data<float>();
  const float* gt_assignment_flat = gt_assignment.data<float>();
  const float* pc_labels_flat = pc_labels.data<float>();
  const float* pc_probs_flat = pc_probs.data<float>();
  const float* pc_count_flat = pc_count.data<float>();
  const float* img_cls_loss_weights_flat = img_cls_loss_weights.data<float>();
  const float* im_labels_flat = im_labels.data<float>();

  auto* bottom_grad = Output(0);
  bottom_grad->ResizeLike(pcl_probs);
  math::Set<float, CPUContext>(bottom_grad->numel(), 0.f,
                               bottom_grad->mutable_data<float>(), &context_);

  float* bottom_grad_flat = bottom_grad->mutable_data<float>();

  int batch_size = pcl_probs.dim32(0);
  int channels = pcl_probs.dim32(1);


    float eps = 1e-5;

    for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < channels; c++) {
            bottom_grad_flat[i * channels + c] = 0;
            if (im_labels_flat[c] != 0) {
                if (c == 0) {
                    if (labels_flat[i] == 0) {
                        bottom_grad_flat[i * channels + c] = -cls_loss_weights_flat[i] 
                            / fmaxf(prob_data_flat[i * channels + c], eps);
                    }
                }
                else {
                    if (labels_flat[i] == c) {
                        int pc_index = gt_assignment_flat[i];
                        if (c != pc_labels_flat[pc_index]) {
                            printf("labels mismatch.\n");
                        }
                        bottom_grad_flat[i * channels + c] = -img_cls_loss_weights_flat[pc_index] 
                            / fmaxf(pc_count_flat[pc_index] * pc_probs_flat[pc_index], eps);
                    }
                }
            }
        }
    }


  math::Scale<float, float, CPUContext>(
      bottom_grad->numel(), float(1.0 / pcl_probs.dim32(0)), bottom_grad_flat,
      bottom_grad_flat, &context_);

  return true;
}

REGISTER_CPU_OPERATOR(PCLLoss, PCLLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(PCLLossGradient, PCLLossGradientOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(PCLLoss)
    .NumInputs(9)
    .NumOutputs(1)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "X", "Input blob of size ")
    .Input(1, "X", "Input blob of size ")
    .Input(2, "X", "Input blob of size ")
    .Input(3, "X", "Input blob of size ")
    .Input(4, "X", "Input blob of size ")
    .Input(5, "X", "Input blob of size ")
    .Input(6, "X", "Input blob of size ")
    .Input(7, "X", "Input blob of size ")
    .Input(8, "X", "Input blob of size ")
    .Output(0, "Y", "Output blob computation");
OPERATOR_SCHEMA(PCLLossGradient).NumInputs(9).NumOutputs(1);

class GetPCLLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PCLLossGradient", "",
        vector<string>{I(0), I(1), I(2), I(3), I(4), I(5), I(6), I(7), I(8)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(PCLLoss, GetPCLLossGradient);

}  // namespace caffe2
