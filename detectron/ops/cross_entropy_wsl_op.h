#ifndef CAFFE2_OPERATORS_CROSS_ENTROPY_WSL_OP_H_
#define CAFFE2_OPERATORS_CROSS_ENTROPY_WSL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LabelCrossEntropyWSLOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LabelCrossEntropyWSLOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ignore_value_(
            this->template GetSingleArgument<float>("ignore_value", -1.0)) {}
  bool RunOnDevice() override;

 protected:
  static constexpr T kLOG_THRESHOLD() { return static_cast<T>(1e-20); }
  float ignore_value_;
  // Input: X, label
  // Output: Y
};

template <typename T, class Context>
class LabelCrossEntropyWSLGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LabelCrossEntropyWSLGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ignore_value_(
            this->template GetSingleArgument<float>("ignore_value", -1.0)) {}
  bool RunOnDevice() override;

 protected:
  // Input: X, label, dY
  // Ouptut: dX. There is no gradient with respect to the label.
  static constexpr T kLOG_THRESHOLD() { return static_cast<T>(1e-20); }
  float ignore_value_;
};

template <typename T, class Context>
class CrossEntropyWithLogitsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CrossEntropyWithLogitsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_mean_(this->template GetSingleArgument<bool>("is_mean", false)) {}
  bool RunOnDevice() override;

 protected:
  // Input: X, label
  // Output: Y
  static constexpr T kLOG_THRESHOLD() { return static_cast<T>(1e-20); }
  bool is_mean_;
};

template <typename T, class Context>
class CrossEntropyWithLogitsGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CrossEntropyWithLogitsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_mean_(this->template GetSingleArgument<bool>("is_mean", false)) {}
  bool RunOnDevice() override;

 protected:
  // Input: X, label, dY
  // Ouptut: dX. There is no gradient with respect to the label.
  static constexpr T kLOG_THRESHOLD() { return static_cast<T>(1e-20); }
  static constexpr T kDIFF_THRESHOLD() { return static_cast<T>(1e+4); }
  bool is_mean_;
};

template <typename T, class Context>
class WeightedCrossEntropyWithLogitsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  WeightedCrossEntropyWithLogitsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_mean_(this->template GetSingleArgument<bool>("is_mean", false)) {}
  bool RunOnDevice() override;

 protected:
  // Input: X, label, weight
  // Output: Y
  static constexpr T kLOG_THRESHOLD() { return static_cast<T>(1e-20); }
  bool is_mean_;
};

template <typename T, class Context>
class WeightedCrossEntropyWithLogitsGradientOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  WeightedCrossEntropyWithLogitsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_mean_(this->template GetSingleArgument<bool>("is_mean", false)) {}
  bool RunOnDevice() override;

 protected:
  // Input: X, label, weight, dY
  // Ouptut: dX. There is no gradient with respect to the label.
  static constexpr T kLOG_THRESHOLD() { return static_cast<T>(1e-20); }
  static constexpr T kDIFF_THRESHOLD() { return static_cast<T>(1e+4); }
  bool is_mean_;
};

template <typename T, class Context>
class SigmoidCrossEntropyWithLogitsWSLOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SigmoidCrossEntropyWithLogitsWSLOp(const OperatorDef& operator_def,
                                     Workspace* ws)
      : Operator<Context>(operator_def, ws),
        log_D_trick_(
            this->template GetSingleArgument<bool>("log_D_trick", false)),
        ignore_value_(
            this->template GetSingleArgument<float>("ignore_value", 0.5)) {}

  bool RunOnDevice() override;

 protected:
  bool log_D_trick_;
  float ignore_value_;
};

template <typename T, class Context>
class SigmoidCrossEntropyWithLogitsWSLGradientOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SigmoidCrossEntropyWithLogitsWSLGradientOp(const OperatorDef& operator_def,
                                             Workspace* ws)
      : Operator<Context>(operator_def, ws),
        log_D_trick_(
            this->template GetSingleArgument<bool>("log_D_trick", false)),
        ignore_value_(
            this->template GetSingleArgument<float>("ignore_value", 0.5)) {}

  bool RunOnDevice() override;

 protected:
  bool log_D_trick_;
  float ignore_value_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CROSS_ENTROPY_WSL_OP_H_
