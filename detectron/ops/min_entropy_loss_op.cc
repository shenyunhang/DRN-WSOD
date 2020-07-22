#include "min_entropy_loss_op.h"

namespace caffe2 {

namespace {}

template <>
bool MinEntropyLossOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& L = Input(1);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(L.dim(), 2);
  CAFFE_ENFORCE_EQ(X.dim32(1), L.dim32(1));

  int N = X.dim32(0);
  int C = X.dim32(1);
  int B = L.dim32(0);

  auto* Y = Output(0);
  Y->Resize(vector<int64_t>());
  math::Set<float, CPUContext>(Y->numel(), 0.f, Y->mutable_data<float>(),
                               &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  auto* Ydata = Y->mutable_data<float>();

  float loss = 0;
  int norm = 0;
  CAFFE_ENFORCE_EQ(L.dim32(0), 1);
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      if (Ldata[c] < 0.5) {
        continue;
      }
      float prob = std::max(Xdata[n * C + c], kLOG_THRESHOLD());
      loss -= (prob * log(prob));
      norm += 1;
    }
  }
  Ydata[0] = loss / norm;

  return true;
}

template <>
bool MinEntropyLossGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& L = Input(1);
  auto& dY = Input(2);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(L.dim(), 2);
  CAFFE_ENFORCE_EQ(X.dim32(1), L.dim32(1));
  CAFFE_ENFORCE_EQ(dY.numel(), 1);

  int N = X.dim32(0);
  int C = X.dim32(1);
  int B = L.dim32(0);

  auto* dX = Output(0);
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                               &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  int norm = 0;
  CAFFE_ENFORCE_EQ(L.dim32(0), 1);
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      if (Ldata[c] < 0.5) {
        continue;
      }
      norm += 1;
    }
  }

  const float scale = dYdata[0] / norm;

  CAFFE_ENFORCE_EQ(L.dim32(0), 1);
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      if (Ldata[c] < 0.5) {
        continue;
      }
      float prob = std::max(Xdata[n * C + c], kLOG_THRESHOLD());
      dXdata[n * C + c] =
          std::min(scale * (-1 + (-1) * float(log(prob))), kDIFF_THRESHOLD());
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(MinEntropyLoss, MinEntropyLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MinEntropyLossGradient,
                      MinEntropyLossGradientOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(MinEntropyLoss)
    .NumInputs(2, 2 + 1)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "X", "Input blob of size N x C")
    .Input(1, "L", "Input Blob of size B x C")
    .Output(0, "Y", "Output blob after computation");
OPERATOR_SCHEMA(MinEntropyLossGradient).NumInputs(3).NumOutputs(1);

class GetMinEntropyLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef("MinEntropyLossGradient", "",
                             vector<string>{I(0), I(1), GO(0)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(MinEntropyLoss, GetMinEntropyLossGradient);

}  // namespace caffe2
