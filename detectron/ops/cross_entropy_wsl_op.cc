#include "cross_entropy_wsl_op.h"

namespace caffe2 {

namespace {}

template <>
bool CrossEntropyWithLogitsOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& L = Input(1);

  // if (InputSize() > 2) {
  // printf("Found unused input in CrossEntropyWithLogits %d\n",
  // InputSize() - 2);
  //}

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(X.sizes(), L.sizes());

  int N = X.dim32(0);
  int C = X.dim32(1);
  float norm = is_mean_ ? C : 1;

  auto* Y = Output(0);
  Y->Resize(vector<int64_t>());
  math::Set<float, CPUContext>(Y->numel(), 0.f, Y->mutable_data<float>(),
                               &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  auto* Ydata = Y->mutable_data<float>();

  float loss = 0.;
  for (int i = 0; i < X.numel(); i++) {
    float prob = std::max(Xdata[i], kLOG_THRESHOLD());
    float one_prob = std::max(1 - Xdata[i], kLOG_THRESHOLD());
    loss -= (Ldata[i] * log(prob) + (1 - Ldata[i]) * log(one_prob));
  }
  Ydata[0] = loss / norm;

  math::Scale<float, float, CPUContext>(Y->numel(), float(1.0 / N), Ydata, Ydata,
                                        &context_);

  return true;
}

template <>
bool CrossEntropyWithLogitsGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& L = Input(1);
  auto& dY = Input(2);

  CAFFE_ENFORCE_EQ(X.numel(), L.numel());
  CAFFE_ENFORCE_EQ(X.dim32(0), L.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), L.dim32(1));
  CAFFE_ENFORCE_EQ(dY.numel(), 1);

  int N = X.dim32(0);
  int C = X.dim32(1);
  float norm = is_mean_ ? C : 1;

  auto* dX = Output(0);
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                               &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  for (int i = 0; i < X.numel(); i++) {
    float grad = dYdata[0];
    float prob = std::max(Xdata[i], kLOG_THRESHOLD());
    float one_prob = std::max(1 - Xdata[i], kLOG_THRESHOLD());
    dXdata[i] = std::min(
        grad * (-1 * Ldata[i] / prob - (-1) * (1 - Ldata[i]) / one_prob) / norm,
        kDIFF_THRESHOLD());
  }

  math::Scale<float, float, CPUContext>(dX->numel(), float(1.0 / N), dXdata, dXdata,
                                        &context_);

  return true;
}

template <>
bool WeightedCrossEntropyWithLogitsOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& L = Input(1);
  const auto& W = Input(2);

  // if (InputSize() > 2) {
  // printf("Found unused input in CrossEntropyWithLogits %d\n",
  // InputSize() - 2);
  //}

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(X.sizes(), L.sizes());
  CAFFE_ENFORCE_EQ(X.sizes(), W.sizes());
  CAFFE_ENFORCE_EQ(X.dim32(0), L.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), L.dim32(1));
  CAFFE_ENFORCE_EQ(X.dim32(0), W.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), W.dim32(1));

  int N = X.dim32(0);
  int C = X.dim32(1);
  float norm = is_mean_ ? C : 1;

  auto* Y = Output(0);
  Y->Resize(vector<int64_t>());
  math::Set<float, CPUContext>(Y->numel(), 0.f, Y->mutable_data<float>(),
                               &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  const float* Wdata = W.data<float>();
  auto* Ydata = Y->mutable_data<float>();

  float loss = 0.;
  for (int i = 0; i < X.numel(); i++) {
    float prob = std::max(Xdata[i], kLOG_THRESHOLD());
    float one_prob = std::max(1 - Xdata[i], kLOG_THRESHOLD());
    loss -= (Ldata[i] * log(prob) + (1 - Ldata[i]) * log(one_prob)) * Wdata[i];
  }
  Ydata[0] = loss / norm;

  math::Scale<float, float, CPUContext>(Y->numel(), float(1.0 / N), Ydata, Ydata,
                                        &context_);

  return true;
}

template <>
bool WeightedCrossEntropyWithLogitsGradientOp<float,
                                              CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& L = Input(1);
  auto& W = Input(2);
  auto& dY = Input(3);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(X.sizes(), L.sizes());
  CAFFE_ENFORCE_EQ(X.sizes(), W.sizes());
  CAFFE_ENFORCE_EQ(X.dim32(0), L.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), L.dim32(1));
  CAFFE_ENFORCE_EQ(X.dim32(0), W.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), W.dim32(1));
  CAFFE_ENFORCE_EQ(dY.numel(), 1);

  int N = X.dim32(0);
  int C = X.dim32(1);
  float norm = is_mean_ ? C : 1;

  auto* dX = Output(0);
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                               &context_);

  const float* Xdata = X.data<float>();
  const float* Ldata = L.data<float>();
  const float* Wdata = W.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  for (int i = 0; i < X.numel(); i++) {
    float grad = dYdata[0];
    float prob = std::max(Xdata[i], kLOG_THRESHOLD());
    float one_prob = std::max(1 - Xdata[i], kLOG_THRESHOLD());
    dXdata[i] = std::min(grad * (-1 * Ldata[i] / prob -
                                 (-1) * (1 - Ldata[i]) / one_prob) / norm,
                         kDIFF_THRESHOLD()) *
                Wdata[i];
  }

  math::Scale<float, float, CPUContext>(dX->numel(), float(1.0 / N), dXdata, dXdata,
                                        &context_);

  return true;
}

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(LabelCrossEntropyWSL)
    .NumInputs(2 + 1)
    .NumOutputs(2)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "X", "$N x C x H x W$")
    .Input(1, "label", "$N x H x W$")
    .Output(0, "Y", "$N$")
    .Output(1, "count", "$N x 2$");
OPERATOR_SCHEMA(LabelCrossEntropyWSLGradient).NumInputs(4).NumOutputs(1);

class GetLabelCrossEntropyWSLGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef("LabelCrossEntropyWSLGradient", "",
                             vector<string>{I(0), I(1), GO(0), O(1)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(LabelCrossEntropyWSL, GetLabelCrossEntropyWSLGradient);

REGISTER_CPU_OPERATOR(CrossEntropyWithLogits,
                      CrossEntropyWithLogitsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(CrossEntropyWithLogitsGradient,
                      CrossEntropyWithLogitsGradientOp<float, CPUContext>);

// addition input for cpg
OPERATOR_SCHEMA(CrossEntropyWithLogits)
    .NumInputs(2, 2 + 1)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "X", "$N x C$")
    .Input(1, "L", "$N x C$")
    .Output(0, "Y", "$1$");
OPERATOR_SCHEMA(CrossEntropyWithLogitsGradient).NumInputs(3).NumOutputs(1);

class GetCrossEntropyWithLogitsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef("CrossEntropyWithLogitsGradient", "",
                             vector<string>{I(0), I(1), GO(0)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(CrossEntropyWithLogits, GetCrossEntropyWithLogitsGradient);

REGISTER_CPU_OPERATOR(WeightedCrossEntropyWithLogits,
                      WeightedCrossEntropyWithLogitsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    WeightedCrossEntropyWithLogitsGradient,
    WeightedCrossEntropyWithLogitsGradientOp<float, CPUContext>);

// addition input for cpg
OPERATOR_SCHEMA(WeightedCrossEntropyWithLogits)
    .Arg("is_mean", "(bool) default to false")
    .NumInputs(3, 3 + 1)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "X", "$N x C$")
    .Input(1, "L", "$N x C$")
    .Input(2, "W", "$N x C$")
    .Output(0, "Y", "$1$");
OPERATOR_SCHEMA(WeightedCrossEntropyWithLogitsGradient)
    .NumInputs(4)
    .NumOutputs(1);

class GetWeightedCrossEntropyWithLogitsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef("WeightedCrossEntropyWithLogitsGradient", "",
                             vector<string>{I(0), I(1), I(2), GO(0)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(WeightedCrossEntropyWithLogits,
                  GetWeightedCrossEntropyWithLogitsGradient);

OPERATOR_SCHEMA(SigmoidCrossEntropyWithLogitsWSL)
    .Arg("log_D_trick", R"DOC(
default is false; if enabled, will use the log d trick to avoid the vanishing
gradients early on; see Goodfellow et. al (2014)
)DOC")
    .Arg("ignore_value", R"DOC(
default is 0.5.
)DOC")
    .NumInputs(2 + 1)
    .NumOutputs(2)
    .IdenticalTypeAndShapeOfInputDim(0, 0)
    .SetDoc(R"DOC(
)DOC")
    .Input(0, "logits", "$N x C x H x W$")
    .Input(1, "targets", "$N x C x H x W$")
    .Output(0, "xentropy", "$N x C$")
    .Output(1, "count", "$N x C x 2$");

OPERATOR_SCHEMA(SigmoidCrossEntropyWithLogitsWSLGradient)
    .NumInputs(4)
    .NumOutputs(1);

struct GetSigmoidCrossEntropyWithLogitsWSLGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    auto log_D_trick = argsHelper.GetSingleArgument<bool>("log_D_trick", false);
    auto ignore_value =
        argsHelper.GetSingleArgument<float>("ignore_value", 0.5);
    return SingleGradientDef(
        "SigmoidCrossEntropyWithLogitsWSLGradient", "",
        vector<string>{GO(0), I(0), I(1), O(1)}, vector<string>{GI(0)},
        vector<Argument>{MakeArgument<bool>("log_D_trick", log_D_trick),
                         MakeArgument<float>("ignore_value", ignore_value)});
  }
};
REGISTER_GRADIENT(SigmoidCrossEntropyWithLogitsWSL,
                  GetSigmoidCrossEntropyWithLogitsWSLGradient);

}  // namespace caffe2
