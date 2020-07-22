#ifndef CAFFE2_OPERATORS_ROI_SCORE_RESHAPE_OP_H_
#define CAFFE2_OPERATORS_ROI_SCORE_RESHAPE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class RoIScoreReshapeOp final : public Operator<Context> {
 public:
  RoIScoreReshapeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        batch_size_(this->template GetSingleArgument<int32_t>("batch_size", 0)),
        num_classes_(
            this->template GetSingleArgument<int32_t>("num_classes", 0)),
        rois_size_(this->template GetSingleArgument<int32_t>("rois_size", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int batch_size_;
  int num_classes_;
  int rois_size_;
};

template <typename T, class Context>
class RoIScoreReshapeGradientOp final : public Operator<Context> {
 public:
  RoIScoreReshapeGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        batch_size_(this->template GetSingleArgument<int32_t>("batch_size", 0)),
        num_classes_(
            this->template GetSingleArgument<int32_t>("num_classes", 0)),
        rois_size_(this->template GetSingleArgument<int32_t>("rois_size", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int batch_size_;
  int num_classes_;
  int rois_size_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ROI_SCORE_RESHAPE_OP_H_
