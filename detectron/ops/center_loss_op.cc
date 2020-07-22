#include <functional>

#include "center_loss_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

// addition input for cpg
OPERATOR_SCHEMA(CenterLoss)
    .NumInputs(6, 6 + 1)
    .NumOutputs(3)
    .SetDoc(R"DOC(
)DOC")
    .Arg("top_k", "(int32_t) default to 10")
    .Arg("update", "(int32_t) default to 128")
    .Arg("lr", "(float) default to 0.5")
    .Arg("display", "(int32_t) default to 1280")
    .Arg("max_iter", "(int32_t) default to 0")
    .Arg("ignore_label", "(int32_t) default to -1")
    .Arg("debug_info", "(bool) default to false")
    .Input(0, "X", "input tensor of size (BxC)")
    .Input(1, "P", "input tensor of size (RxC)")
    .Input(2, "F", "input tensor of size (RxD)")
    .Input(3, "CF", "input tensor of size (CxMxD)")
    .Input(4, "dCF", "input tensor of size (CxMxD)")
    .Input(5, "ndCF", "input tensor of size (CxM)")
    .Output(0, "L", "output tensor of size (1)")
    .Output(1, "D", "output tensor of size (CxKxD)")
    .Output(2, "S", "output tensor of size (C)");

OPERATOR_SCHEMA(CenterLossGradient)
    .NumInputs(8)
    .NumOutputs(4)
    .AllowInplace({{2, 1}, {3, 2}, {4, 3}});

namespace {

class GetCenterLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    auto top_k = argsHelper.GetSingleArgument<int32_t>("top_k", 10);
    auto update = argsHelper.GetSingleArgument<int32_t>("update", 128);
    auto lr = argsHelper.GetSingleArgument<float>("lr", 0.5);
    auto display = argsHelper.GetSingleArgument<int32_t>("display", 1280);
    auto max_iter = argsHelper.GetSingleArgument<int32_t>("max_iter", 0);
    auto ignore_label =
        argsHelper.GetSingleArgument<int32_t>("ignore_label", -1);
    auto debug_info = argsHelper.GetSingleArgument<bool>("debug_info", false);

    return SingleGradientDef(
        "CenterLossGradient", "",
        vector<string>{I(0), I(1), I(3), I(4), I(5), O(1), O(2), GO(0)},
        vector<string>{GI(2), I(3), I(4), I(5)});
    // vector<Argument>{MakeArgument<int32_t>("top_k", top_k),
    // MakeArgument<int32_t>("update", update),
    // MakeArgument<float>("lr", lr),
    // MakeArgument<int32_t>("display", display),
    // MakeArgument<int32_t>("max_iter", max_iter),
    // MakeArgument<int32_t>("ignore_label", ignore_label),
    // MakeArgument<bool>("debug_info", debug_info)});
  }
};

REGISTER_GRADIENT(CenterLoss, GetCenterLossGradient);

}  // namespace

}  // namespace caffe2
