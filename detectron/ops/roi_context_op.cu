#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "roi_context_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ROIContextForward(const int nthreads, const T* rois,
                                  const float context_ratio_, const int max_h,
                                  const int max_w, T* Frois, T* Crois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;

    const T* offset_rois = rois + n * 5;
    T* offset_Frois = Frois + n * 9;
    T* offset_Crois = Crois + n * 9;

    // x1 y1 x2 y2
    float x1 = offset_rois[1];
    float y1 = offset_rois[2];
    float x2 = offset_rois[3];
    float y2 = offset_rois[4];

    float rois_w = x2 - x1;
    float rois_h = y2 - y1;

    float rois_inner_w = rois_w / context_ratio_;
    float rois_inner_h = rois_h / context_ratio_;

    float rois_outer_w = rois_w * context_ratio_;
    float rois_outer_h = rois_h * context_ratio_;

    float inner_residual_w = rois_w - rois_inner_w;
    float inner_residual_h = rois_h - rois_inner_h;

    float outer_residual_w = rois_outer_w - rois_w;
    float outer_residual_h = rois_outer_h - rois_h;

    offset_Frois[0] = offset_rois[0];
    offset_Frois[1] = offset_rois[1];
    offset_Frois[2] = offset_rois[2];
    offset_Frois[3] = offset_rois[3];
    offset_Frois[4] = offset_rois[4];
    offset_Frois[5] = offset_rois[1];
    offset_Frois[6] = offset_rois[2];
    offset_Frois[7] = offset_rois[3];
    offset_Frois[8] = offset_rois[4];

    offset_Crois[0] = offset_rois[0];
    offset_Crois[1] = offset_rois[1];
    offset_Crois[2] = offset_rois[2];
    offset_Crois[3] = offset_rois[3];
    offset_Crois[4] = offset_rois[4];
    offset_Crois[5] = offset_rois[1];
    offset_Crois[6] = offset_rois[2];
    offset_Crois[7] = offset_rois[3];
    offset_Crois[8] = offset_rois[4];

    offset_Frois[5] += inner_residual_w / 2;
    offset_Frois[6] += inner_residual_h / 2;
    offset_Frois[7] -= inner_residual_w / 2;
    offset_Frois[8] -= inner_residual_h / 2;

    offset_Crois[1] -= outer_residual_w / 2;
    offset_Crois[2] -= outer_residual_h / 2;
    offset_Crois[3] += outer_residual_w / 2;
    offset_Crois[4] += outer_residual_h / 2;

    offset_Frois[5] = min(max(offset_Frois[5], T(0)), T(max_w));
    offset_Frois[6] = min(max(offset_Frois[6], T(0)), T(max_h));
    offset_Frois[7] = min(max(offset_Frois[7], T(0)), T(max_w));
    offset_Frois[8] = min(max(offset_Frois[8], T(0)), T(max_h));

    offset_Crois[1] = min(max(offset_Crois[1], T(0)), T(max_w));
    offset_Crois[2] = min(max(offset_Crois[2], T(0)), T(max_h));
    offset_Crois[3] = min(max(offset_Crois[3], T(0)), T(max_w));
    offset_Crois[4] = min(max(offset_Crois[4], T(0)), T(max_h));
  }
}

}  // namespace

template <>
bool RoIContextOp<float, CUDAContext>::RunOnDevice() {
  auto& R = Input(0);
  auto& X = Input(1);
  auto* RF = Output(0);
  auto* RC = Output(1);

  const int num_rois = R.dim32(0);
  const int num_channels = R.dim32(1);

  CAFFE_ENFORCE_EQ(num_channels, 5);

  RF->Resize(num_rois, 9);
  RC->Resize(num_rois, 9);

  ROIContextForward<float><<<CAFFE_GET_BLOCKS(num_rois), CAFFE_CUDA_NUM_THREADS,
                             0, context_.cuda_stream()>>>(
      num_rois, R.data<float>(), context_ratio_, X.dim32(2), X.dim32(3),
      RF->template mutable_data<float>(), RC->template mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(RoIContext, RoIContextOp<float, CUDAContext>);

}  // namespace caffe2
