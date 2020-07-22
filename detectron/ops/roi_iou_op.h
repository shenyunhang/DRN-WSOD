#ifndef CAFFE2_OPERATORS_ROI_IOU_OP_H_
#define CAFFE2_OPERATORS_ROI_IOU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class RoIIoUOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RoIIoUOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)) {}
  ~RoIIoUOp() {}

  bool RunOnDevice() override;

 protected:
  bool debug_info_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ROI_IOU_OP_H_
