#ifndef ROI_CONTEXT_OP_H_
#define ROI_CONTEXT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class RoIContextOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit RoIContextOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        context_ratio_(
            this->template GetSingleArgument<float>("context_ratio", 1.8)) {
    CAFFE_ENFORCE_GT(context_ratio_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float context_ratio_;
};

}  // namespace caffe2

#endif  // ROI_CONTEXT_OP_H_
