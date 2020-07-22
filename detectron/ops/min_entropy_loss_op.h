#ifndef CAFFE2_OPERATORS_MIN_ENTROPY_LOSS_OP_H_
#define CAFFE2_OPERATORS_MIN_ENTROPY_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class MinEntropyLossOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MinEntropyLossOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X, label
  // Output: Y
  static constexpr T kLOG_THRESHOLD() { return static_cast<T>(1e-20); }
};

template <typename T, class Context>
class MinEntropyLossGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MinEntropyLossGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X, label, dY
  // Ouptut: dX. There is no gradient with respect to the label.
  static constexpr T kLOG_THRESHOLD() { return static_cast<T>(1e-20); }
  static constexpr T kDIFF_THRESHOLD() { return static_cast<T>(1e+4); }
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_MIN_ENTROPY_LOSS_OP_H_
