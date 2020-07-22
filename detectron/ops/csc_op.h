#ifndef CAFFE2_OPERATORS_CSC_OP_H_
#define CAFFE2_OPERATORS_CSC_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

const float kMIN_SCORE = -1.0 * 1e20;

template <typename T, class Context>
class CSCOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CSCOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        tau_(this->template GetSingleArgument<float>("tau", 0.7)),
        max_iter_(this->template GetSingleArgument<int>("max_iter", 0)),
        cur_iter_(0),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)),
        fg_threshold_(
            this->template GetSingleArgument<float>("fg_threshold", 0.1)),
        mass_threshold_(
            this->template GetSingleArgument<float>("mass_threshold", 0.2)),
        density_threshold_(
            this->template GetSingleArgument<float>("density_threshold", 0.0)),
        area_sqrt_(this->template GetSingleArgument<bool>("area_sqrt", true)),
        context_scale_(
            this->template GetSingleArgument<float>("context_scale", 1.8)) {}
  ~CSCOp() {}

  bool RunOnDevice() override;

 protected:
  float tau_;
  int max_iter_;
  int cur_iter_;
  float fg_threshold_;
  float mass_threshold_;
  float density_threshold_;
  bool area_sqrt_;
  float context_scale_;

  bool debug_info_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CSC_OP_H_
