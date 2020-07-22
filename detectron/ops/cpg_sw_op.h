#ifndef CAFFE2_OPERATORS_CPG_SW_OP_H_
#define CAFFE2_OPERATORS_CPG_SW_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class CPGSWOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CPGSWOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        tau_(this->template GetSingleArgument<float>("tau", 0.7)),
        min_loss_(this->template GetSingleArgument<float>("min_loss", 0.1)),
        max_iter_(this->template GetSingleArgument<int>("max_iter", 0)),
        cur_iter_(0),
        acm_loss_(0),
        acm_cnt_(0),
        acm_sw_loss_(0),
        acm_sw_cnt_(0),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)) {}
  ~CPGSWOp() {}

  bool RunOnDevice() override;

 protected:
  float min_loss_;
  float tau_;
  int max_iter_;
  int cur_iter_;

  float acm_loss_;
  int acm_cnt_;
  float acm_sw_loss_;
  int acm_sw_cnt_;

  bool debug_info_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CPG_SW_OP_H_
