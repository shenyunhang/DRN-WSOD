#include "roi_context_op.h"

#include <cfloat>

namespace caffe2 {

OPERATOR_SCHEMA(RoIContext)
    .NumInputs(2)
    .NumOutputs({2})
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      const TensorShape& X = in[0];
      const int num_rois = X.dims(0);
      const int num_channels = X.dims(1);
      TensorShape Y = CreateTensorShape(vector<int>({num_rois, num_channels}),
                                        X.data_type());
      return vector<TensorShape>({Y});
    })
    .SetDoc(R"DOC(
)DOC")
    .Arg("context_ratio", "ONTEXT_RATIO. ")
    .Input(0, "R", "The input 2-D tensor of rois.")
    .Input(1, "X", "The input 4-D tensor of rois.")
    .Output(0, "RF", "RoI Frame.")
    .Output(1, "RC", "RoI Context.");

NO_GRADIENT(RoIContext);

}  // namespace caffe2
