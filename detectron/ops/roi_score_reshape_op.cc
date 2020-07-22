#include "roi_score_reshape_op.h"

namespace caffe2 {

// Implementation for the CPU context.
template <>
bool RoIScoreReshapeOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& R = Input(1);
  auto* Y = Output(0);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(X.dim32(0), R.dim32(0));
  CAFFE_ENFORCE_EQ(X.dim32(1), num_classes_);

  CAFFE_ENFORCE_EQ(R.dim(), 2);
  CAFFE_ENFORCE_EQ(R.dim32(0), X.dim32(0));
  CAFFE_ENFORCE_EQ(R.dim32(1), 5);

  Y->Resize(batch_size_, num_classes_, rois_size_, 1);
  math::Set<float, CPUContext>(Y->numel(), 0.f, Y->mutable_data<float>(),
                               &context_);

  const int N = X.dim32(0);
  const float* Xdata = X.data<float>();
  const float* Rdata = R.data<float>();
  float* Ydata = Y->mutable_data<float>();

  int b = -1;
  int r = 0;
  for (int n = 0; n < N; n++) {
    if (b != Rdata[n * 5 + 0]) {
      r = 0;
      b = Rdata[n * 5 + 0];
    }
    for (int c = 0; c < num_classes_; c++) {
      int Xidx = n * num_classes_ + c;
      int Yidx = ((b * num_classes_) + c) * rois_size_ + r;
      Ydata[Yidx] = Xdata[Xidx];
    }
    r++;
  }

  return true;
}

// Implementation for the CPU context.
template <>
bool RoIScoreReshapeGradientOp<float, CPUContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto& R = Input(1);
  auto* dX = Output(0);

  CAFFE_ENFORCE_EQ(dY.dim(), 4);
  CAFFE_ENFORCE_EQ(dY.dim32(0), batch_size_);
  CAFFE_ENFORCE_EQ(dY.dim32(1), num_classes_);
  CAFFE_ENFORCE_EQ(dY.dim32(2), rois_size_);
  CAFFE_ENFORCE_EQ(dY.dim32(3), 1);

  CAFFE_ENFORCE_EQ(R.dim(), 2);
  CAFFE_ENFORCE_EQ(R.dim32(1), 5);

  dX->Resize(R.dim32(0), num_classes_);
  math::Set<float, CPUContext>(dX->numel(), 0.f, dX->mutable_data<float>(),
                               &context_);

  const int N = R.dim32(0);
  const float* Rdata = R.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  int b = -1;
  int r = 0;
  for (int n = 0; n < N; n++) {
    if (b != Rdata[n * 5 + 0]) {
      r = 0;
      b = Rdata[n * 5 + 0];
    }
    for (int c = 0; c < num_classes_; c++) {
      int dXidx = n * num_classes_ + c;
      int dYidx = ((b * num_classes_) + c) * rois_size_ + r;
      dXdata[dXidx] = dYdata[dYidx];
    }
    r++;
  }

  return true;
}

REGISTER_CPU_OPERATOR(RoIScoreReshape, RoIScoreReshapeOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(RoIScoreReshapeGradient,
                      RoIScoreReshapeGradientOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(RoIScoreReshape)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
)DOC")
    .Arg("axis",
         "(int) default to 1; describes the axis of the inputs when coerced "
         "to 2D; defaults to one because the 0th axis most likely describes "
         "the batch_size")
    .Input(0, "input", "The input tensor into a 2D matrix of size (NxC).")
    .Input(1, "input", "The roi tensor into a 2D matrix of size (Nx5).")
    .Output(0, "output", "The output tensor of size (BxCxHx1).");

// Input: Y, dY. Output: dX
OPERATOR_SCHEMA(RoIScoreReshapeGradient).NumInputs(2).NumOutputs(1);

namespace {

class GetRoIScoreReshapeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(def_.type() + "Gradient", "",
                             vector<string>{GO(0), I(1)},
                             vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(RoIScoreReshape, GetRoIScoreReshapeGradient);

}  // namespace

}  // namespace caffe2
