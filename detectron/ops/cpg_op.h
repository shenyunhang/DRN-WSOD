#ifndef CAFFE2_OPERATORS_CPG_OP_H_
#define CAFFE2_OPERATORS_CPG_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class CPGOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CPGOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        gWorkspace_(ws),
        tau_(this->template GetSingleArgument<float>("tau", 0.7)),
        max_iter_(this->template GetSingleArgument<int>("max_iter", 0)),
        cur_iter_(0),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)),
        cpg_net_name_(
            this->template GetSingleArgument<string>("cpg_net_name", "cpg")),
        data_blob_name_(
            this->template GetSingleArgument<string>("data_blob_name", "data")),
        pred_blob_name_(this->template GetSingleArgument<string>(
            "pred_blob_name", "cls_prob")) {}
  ~CPGOp() {}

  bool RunOnDevice() override;

 protected:
  Workspace* gWorkspace_;

  float tau_;
  int max_iter_;
  int cur_iter_;
  string cpg_net_name_;
  string pred_blob_name_;
  string data_blob_name_;

  bool debug_info_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_CPG_OP_H_
