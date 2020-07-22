#include "acm_weightdecay_momentum_sgd_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ACMWeightDecayMomentumSGDUpdate,
                      ACMWeightDecayMomentumSGDUpdateOp<float, CPUContext>);
OPERATOR_SCHEMA(ACMWeightDecayMomentumSGDUpdate)
    .NumInputs(5)
    .NumOutputs(4)
    .AllowInplace({{0, 0}, {1, 1}, {3, 2}, {4, 3}})
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(4);
      out[0] = in[0];
      out[1] = in[1];
      out[2] = in[3];
      out[3] = in[4];
      return out;
    })
    .SetDoc(R"DOC(
)DOC");
SHOULD_NOT_DO_GRADIENT(ACMWeightDecayMomentumSGDUpdate);
}
