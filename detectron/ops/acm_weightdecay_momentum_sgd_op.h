#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename Context>
void momentum_sgd_update_mult(const int N, const float* g, const float* m,
                              float* ng, float* nm, const float* lr,
                              const float lr_mult, const float momentum,
                              const bool nesterov, float* param,
                              Context* /*context*/) {
  const float LR = lr[0] * lr_mult;
  for (auto i = 0; i < N; ++i) {
    if (!nesterov) {
      const float adjusted_gradient = LR * g[i] + momentum * m[i];
      nm[i] = adjusted_gradient;
      ng[i] = adjusted_gradient;
    } else {
      const float mi = m[i];
      const float mi_new = momentum * mi + LR * g[i];
      nm[i] = mi_new;
      ng[i] = (1 + momentum) * mi_new - momentum * mi;
    }

    if (param) {
      param[i] -= ng[i];
    }
  }
}

template <typename T, class Context>
class ACMWeightDecayMomentumSGDUpdateOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ACMWeightDecayMomentumSGDUpdateOp(const OperatorDef& operator_def,
                                    Workspace* ws)
      : Operator<Context>(operator_def, ws),
        momentum_(this->template GetSingleArgument<T>("momentum", 0.0)),
        nesterov_(this->template GetSingleArgument<int>("nesterov", 0)),
        weight_decay_(this->template GetSingleArgument<T>("weight_decay", 0.0)),
        iter_size_(this->template GetSingleArgument<int>("iter_size", 1)),
        gpu_num_(this->template GetSingleArgument<int>("gpu_num", 1)),
        iter_count_(0),
        lr_mult_(this->template GetSingleArgument<T>("lr_mult", 1.0)) {}

  bool RunOnDevice() override {
    auto device_type = Context::GetDeviceType();
    // Iter live on the CPU
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(GRAD, device_type));
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(ACMGRAD, device_type));
    CAFFE_ENFORCE(OperatorBase::InputIsTensorType(MOMENTUM, device_type));
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(MOMENTUM).numel());
    CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(ACMGRAD).numel());
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    Output(OUTPUT_ACMGRAD)->ResizeLike(Input(ACMGRAD));
    Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));

    // Init
    if (iter_count_ == 0) {
      math::Set<T, Context>(Input(PARAM).numel(), 0.0,
                            Output(OUTPUT_ACMGRAD)->template mutable_data<T>(),
                            &context_);
      math::Set<T, Context>(Input(PARAM).numel(), 0.0,
                            Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
                            &context_);
    }

    // ACM Grad
    math::Add<T, Context>(Input(PARAM).numel(), Input(GRAD).template data<T>(),
                          Input(ACMGRAD).template data<T>(),
                          Output(OUTPUT_ACMGRAD)->template mutable_data<T>(),
                          &context_);

    iter_count_ += 1;

    if (iter_count_ % iter_size_ == 0) {
      // Normalize() in Caffe
      math::Scale<T, T, Context>(
          Input(PARAM).numel(), T(1.0 / (iter_size_ * gpu_num_)),
          Output(OUTPUT_ACMGRAD)->template data<T>(),
          Output(OUTPUT_ACMGRAD)->template mutable_data<T>(), &context_);

      // Regularize() in Caffe
      // add weigt decay
      math::Axpy<T, T, Context>(
          Input(PARAM).numel(), weight_decay_, Input(PARAM).template data<T>(),
          Output(OUTPUT_ACMGRAD)->template mutable_data<T>(), &context_);

      // Get local rate

      // ComputeUpdateValue() in Caffe
      // Compute the update to history, the copy it to the parameter diff
      momentum_sgd_update_mult<Context>(
          Input(PARAM).numel(), Output(OUTPUT_ACMGRAD)->template data<T>(),
          Input(MOMENTUM).template data<T>(),
          Output(OUTPUT_ACMGRAD)->template mutable_data<T>(),
          Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
          Input(LR).template data<T>(), lr_mult_, momentum_, nesterov_,
          Output(OUTPUT_PARAM)->template mutable_data<T>(), &context_);

      // ClearParamDIffs() in Caffe
      // zero-init the params
      math::Set<T, Context>(Input(PARAM).numel(), 0.0,
                            Output(OUTPUT_ACMGRAD)->template mutable_data<T>(),
                            &context_);
    }

    return true;
  }

 protected:
  T momentum_{0.9};
  bool nesterov_;
  INPUT_TAGS(GRAD, MOMENTUM, LR, PARAM, ACMGRAD);
  OUTPUT_TAGS(OUTPUT_GRAD, OUTPUT_MOMENTUM, OUTPUT_PARAM, OUTPUT_ACMGRAD);

  T weight_decay_;
  int iter_size_;
  int gpu_num_;
  int iter_count_;

  T lr_mult_;
};
}  // namespace caffe2
