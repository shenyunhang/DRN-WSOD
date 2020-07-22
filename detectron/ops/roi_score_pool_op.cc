#include <functional>

#include "roi_score_pool_op.h"

namespace caffe2 {

template <>
bool RoIScorePoolOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const int batch_size = X.dim32(0);

  auto* Y = Output(0);
  Y->Resize(batch_size, num_classes_);
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(Y->numel(), 0.f, Ydata, &context_);

  for (int i = 0; i < InputSize(); ++i) {
    const auto& X = Input(i);
    const float* Xdata = X.data<float>();
    const int channels = X.dim32(1);
    // const int height = X.dim32(2);
    // const int width = X.dim32(3);
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
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < channels; ++c) {
        int c_Y = c % num_classes_;
        int index_Y = b * num_classes_ + c_Y;
        for (int w = 0; w < width; ++w) {
          for (int h = 0; h < height; ++h) {
            int index_X = ((b * channels + c) * width + w) * height + h;
            Ydata[index_Y] += Xdata[index_X];
          }
        }
      }
    }
  }
  return true;
}

template <>
bool RoIScorePoolGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const float* dYdata = dY.data<float>();

  for (int i = 1; i < InputSize(); ++i) {
    const auto& X = Input(i);
    // TODO: Handle the storage_order properly to get the NCWH.
    int batch_size = X.dim32(0);
    int channels = X.dim32(1);
    // int height = X.dim32(2);
    // int width = X.dim32(3);
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

    auto* dX = Output(i - 1);
    dX->ResizeLike(X);
    float* dXdata = dX->mutable_data<float>();
    math::Set<float, CPUContext>(dX->numel(), 0.f, dXdata, &context_);

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < channels; ++c) {
        int c_dY = c % num_classes_;
        int index_dY = b * num_classes_ + c_dY;
        for (int w = 0; w < width; ++w) {
          for (int h = 0; h < height; ++h) {
            int index_dX = ((b * channels + c) * width + w) * height + h;
            dXdata[index_dX] = dYdata[index_dY];
          }
        }
      }
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(RoIScorePool, RoIScorePoolOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(RoIScorePoolGradient,
                      RoIScorePoolGradientOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(RoIScorePool)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
)DOC")
    .Arg("num_classes_", "(int32_t) default to -1")
    .Output(0, "Y", "output tensor of size (NxC)");

OPERATOR_SCHEMA(RoIScorePoolGradient)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX);

namespace {

class GetRoIScorePoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    if (GradOut(0).IsEmpty()) {
      return {};
    }
    vector<string> ins;
    ins.push_back(GO(0));
    for (int i = 0; i < def_.input_size(); ++i) {
      ins.push_back(I(i));
    }
    vector<string> outs;
    for (int i = 0; i < def_.input_size(); ++i) {
      outs.push_back(GI(i));
    }
    return SingleGradientDef("RoIScorePoolGradient", "", ins, outs);
  }
};

REGISTER_GRADIENT(RoIScorePool, GetRoIScorePoolGradient);

}  // namespace

}  // namespace caffe2
