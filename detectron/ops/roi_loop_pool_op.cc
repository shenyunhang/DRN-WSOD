#include "roi_loop_pool_op.h"

#include <cfloat>

namespace caffe2 {

using std::max;
using std::min;


// Input: X, rois
// Output case #1: Y, argmaxes (train mode)
// Output case #2: Y           (test mode)
OPERATOR_SCHEMA(RoILoopPool)
    .NumInputs(2)
    .NumOutputs({1, 2})
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      const StorageOrder order = StringToStorageOrder(
          helper.GetSingleArgument<string>("order", "NCHW"));
      const TensorShape& X = in[0];
      const int num_channels =
          (order == StorageOrder::NCHW ? X.dims(1) : X.dims(3));
      const TensorShape& R = in[1];
      const int num_rois = R.dims(0);
      const int pooled_height = helper.GetSingleArgument<int>("pooled_h", 1);
      const int pooled_width = helper.GetSingleArgument<int>("pooled_w", 1);
      TensorShape Y = CreateTensorShape(
          vector<int>({num_rois, num_channels, pooled_height, pooled_width}),
          X.data_type());

      bool is_test = helper.GetSingleArgument<int>(OpSchema::Arg_IsTest, 0);
      if (!is_test) {
        TensorShape argmaxes = Y;
        argmaxes.set_data_type(TensorProto_DataType_INT32);
        return vector<TensorShape>({Y, argmaxes});
      } else {
        return vector<TensorShape>({Y});
      }
    })
    .SetDoc(R"DOC(
Carries out ROI Pooling for Faster-RCNN.
Depending on the mode, there are multiple output cases:

  Output case #1: Y, argmaxes (train mode)
  Output case #2: Y           (test mode)
)DOC")
    .Arg(
        "is_test",
        "If set, run in test mode and skip computation of argmaxes (used for "
        "gradient computation). Only one output tensor is produced. "
        "(Default: false).")
    .Arg("order", "A StorageOrder string (Default: \"NCHW\").")
    .Arg("pooled_h", "The pooled output height (Default: 1).")
    .Arg("pooled_w", "The pooled output width (Default: 1).")
    .Arg(
        "spatial_scale",
        "Multiplicative spatial scale factor to translate ROI coords from "
        "their input scale to the scale used when pooling (Default: 1.0).")
    .Input(
        0,
        "X",
        "The input 4-D tensor of data. Only NCHW order is currently supported.")
    .Input(
        1,
        "rois",
        "RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of "
        "shape (num_rois, 9) given as [[batch_id, x1, y1, x2, y2], ...].")
    .Output(
        0,
        "Y",
        "RoI pooled output 4-D tensor of shape "
        "(num_rois, channels, pooled_h, pooled_w).")
    .Output(
        1,
        "argmaxes",
        "Argmaxes corresponding to indices in X used for gradient computation. "
        "Only output if arg \"is_test\" is false.");

// Input: X, rois, argmaxes, dY (aka "gradOutput")
// Output: dX (aka "gradInput")
OPERATOR_SCHEMA(RoILoopPoolGradient).NumInputs(4).NumOutputs(1);

class GetRoILoopPoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RoILoopPoolGradient",
        "",
        vector<string>{I(0), I(1), O(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(RoILoopPool, GetRoILoopPoolGradient);

} // namespace caffe2
