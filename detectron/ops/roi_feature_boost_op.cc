#include <functional>

#include "roi_feature_boost_op.h"

namespace caffe2 {

template <>
bool RoIFeatureBoostOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& S = Input(1);

  CAFFE_ENFORCE_EQ(S.dim32(0), S.numel());
  CAFFE_ENFORCE_EQ(X.dim32(0), S.dim32(0));

  const int batch_size = X.dim32(0);
  const int feature_size = X.size_from_dim(1);

  const float* Xdata = X.data<float>();
  const float* Sdata = S.data<float>();

  auto* Y = Output(0);
  Y->ResizeLike(X);
  float* Ydata = Y->mutable_data<float>();

  for (int b = 0; b < batch_size; ++b) {
    int index_S = b;
    for (int f = 0; f < feature_size; ++f) {
      int index_XY = b * feature_size + f;
      Ydata[index_XY] = Xdata[index_XY] * Sdata[index_S];
    }
  }

  return true;
}

template <>
bool RoIFeatureBoostGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& S = Input(1);

  CAFFE_ENFORCE_EQ(S.dim32(0), S.numel());
  CAFFE_ENFORCE_EQ(dY.dim32(0), S.dim32(0));

  const int batch_size = dY.dim32(0);
  const int feature_size = dY.size_from_dim(1);

  const float* dYdata = dY.data<float>();
  const float* Sdata = S.data<float>();

  auto* dX = Output(0);
  dX->ResizeLike(dY);
  float* dXdata = dX->mutable_data<float>();

  for (int b = 0; b < batch_size; ++b) {
    int index_S = b;
    for (int f = 0; f < feature_size; ++f) {
      int index_dXY = b * feature_size + f;

      dXdata[index_dXY] = dYdata[index_dXY] * Sdata[index_S];
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(RoIFeatureBoost, RoIFeatureBoostOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(RoIFeatureBoostGradient,
                      RoIFeatureBoostGradientOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(RoIFeatureBoost)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "X", "The input tensor of size (NxCxHxW).")
    .Input(1, "S", "The roi tensor of size (N).")
    .Output(0, "Y", "output tensor of size (NxCxHxW).");

OPERATOR_SCHEMA(RoIFeatureBoostGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

namespace {

class GetRoIFeatureBoostGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(def_.type() + "Gradient", "",
                             vector<string>{GO(0), I(1)},
                             vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(RoIFeatureBoost, GetRoIFeatureBoostGradient);

}  // namespace

}  // namespace caffe2
