#include <functional>

#include "caffe2/core/context_gpu.h"
#include "roi_iou_op.h"

namespace caffe2 {

namespace {

//# compute overlaps
//# intersection
// ixmin = np.maximum(BBGT[:, 0], bb[0])
// iymin = np.maximum(BBGT[:, 1], bb[1])
// ixmax = np.minimum(BBGT[:, 2], bb[2])
// iymax = np.minimum(BBGT[:, 3], bb[3])
// iw = np.maximum(ixmax - ixmin + 1., 0.)
// ih = np.maximum(iymax - iymin + 1., 0.)
// inters = iw * ih

//# union
// uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
//(BBGT[:, 2] - BBGT[:, 0] + 1.) *
//(BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

// overlaps = inters / uni

template <typename T>
__global__ void iou(const int nthreads, const T* Rdata, const int n, T* Jdata) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int i = idx % n;
    int j = idx / n;
    if (i == j) {
      Jdata[idx] = 1.0;
      continue;
    }

    int ixmin = Rdata[i * 5 + 1];
    int iymin = Rdata[i * 5 + 2];
    int ixmax = Rdata[i * 5 + 3];
    int iymax = Rdata[i * 5 + 4];

    int jxmin = Rdata[j * 5 + 1];
    int jymin = Rdata[j * 5 + 2];
    int jxmax = Rdata[j * 5 + 3];
    int jymax = Rdata[j * 5 + 4];

    int xmin = max(ixmin, jxmin);
    int ymin = max(iymin, jymin);
    int xmax = min(ixmax, jxmax);
    int ymax = min(iymax, jymax);

    int w = max(xmax - xmin + 1., 0.);
    int h = max(ymax - ymin + 1., 0.);
    float inters = w * h;

    float uni = (ixmax - ixmin + 1.) * (iymax - iymin + 1.) +
                (jxmax - jxmin + 1.) * (jymax - jymin + 1.) - inters;

    float iou = inters / uni;
    Jdata[idx] = iou;
  }
}

}  // namespace

template <>
bool RoIIoUOp<float, CUDAContext>::RunOnDevice() {
  const auto& R = Input(0);

  CAFFE_ENFORCE_EQ(R.dim(), 2);
  CAFFE_ENFORCE_EQ(R.dim32(1), 5);

  const int n = R.dim32(0);

  auto* J = Output(0);

  J->Resize(n, n);

  iou<float><<<CAFFE_GET_BLOCKS(n * n), CAFFE_CUDA_NUM_THREADS, 0,
               context_.cuda_stream()>>>(n * n, R.data<float>(), n,
                                         J->mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(RoIIoU, RoIIoUOp<float, CUDAContext>);

}  // namespace caffe2
