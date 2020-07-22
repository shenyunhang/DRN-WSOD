#include <functional>

#include <cfloat>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include "roi_label_op.h"

namespace caffe2 {

template <>
bool RoILabelOp<float, CPUContext>::RunOnDevice() {
  const auto& S = Input(0);
  const auto& U = Input(1);
  const auto& L = Input(2);

  CAFFE_ENFORCE_EQ(S.dim(), 2);
  CAFFE_ENFORCE_EQ(U.dim(), 2);
  CAFFE_ENFORCE_EQ(L.dim(), 2);
  CAFFE_ENFORCE_EQ(S.dim32(0), U.dim32(0));
  CAFFE_ENFORCE((S.dim32(1) == L.dim32(1)) || (S.dim32(1) == L.dim32(1) + 1));
  CAFFE_ENFORCE_EQ(U.dim32(0), U.dim32(1));
  CAFFE_ENFORCE_EQ(L.dim32(0), 1);

  const int num_roi = S.dim32(0);
  const int num_class_s = S.dim32(1);
  const int num_class = L.dim32(1);
  const int offset_class = num_class_s - num_class;

  auto* RL = Output(0);
  auto* RW = Output(1);

  RL->Resize(num_roi);
  RW->Resize(num_roi);

  const float* Sdata = S.data<float>();
  const float* Udata = U.data<float>();
  const float* Ldata = L.data<float>();

  const float* CWdata = (InputSize() > 3 ? Input(3).data<float>() : nullptr);

  int* RLdata = RL->mutable_data<int>();
  float* RWdata = RW->mutable_data<float>();

  vector<int> highest_n;
  vector<int> highest_c;
  vector<float> highest_p;

  for (int c = 0; c < num_class; c++) {
    if (Ldata[c] == 1) {
    } else {
      continue;
    }

    for (int k = 0; k < top_k_; k++) {
      float max_pred = -FLT_MAX;
      int max_idx = -1;
      for (int n = 0; n < num_roi; n++) {
        if (max_pred < Sdata[n * num_class_s + c + offset_class]) {
          if (std::find(highest_n.begin(), highest_n.end(), n) !=
              highest_n.end()) {
          } else {
            max_pred = Sdata[n * num_class_s + c + offset_class];
            max_idx = n;
          }
        }
      }

      highest_n.push_back(max_idx);
      highest_c.push_back(c);
      highest_p.push_back(max_pred);
    }
  }

  std::srand(unsigned(std::time(0)));
  std::vector<int> myvector;

  // set some values:
  for (int n = 0; n < num_roi; n++) {
    myvector.push_back(n);  // 1 2 3 4 5 6 7 8 9
  }
  // using built-in random generator:
  std::random_shuffle(myvector.begin(), myvector.end());

  int num_pos = 0;
  int num_neg = 0;

  // for (int n = 0; n < num_roi; n++) {
  for (std::vector<int>::iterator it = myvector.begin(); it != myvector.end();
       ++it) {
    int n = *it;
    float max_iou = -FLT_MAX;
    int max_idx = -1;
    for (int i = 0; i < highest_n.size(); i++) {
      int g = highest_n[i];
      if (max_iou < Udata[n * num_roi + g]) {
        max_iou = Udata[n * num_roi + g];
        max_idx = i;
      }
    }

    int assign_n = highest_n[max_idx];
    int assign_c = highest_c[max_idx];
    float assign_w = CWdata ? CWdata[assign_c] : highest_p[max_idx];

    if (max_iou >= fg_thresh_ && num_pos <= num_pos_) {
      assign_c = assign_c + 1;
      num_pos++;
      acc_fg_rois_++;
      acc_fg_weight_ += assign_w;
    } else if (max_iou >= bg_thresh_lo_ && max_iou < bg_thresh_hi_ &&
               num_neg <= num_neg_) {
      assign_c = 0;
      num_neg++;
      acc_bg_rois_++;
      acc_bg_weight_ += assign_w;
    } else {
      assign_c = assign_c + 1;
      assign_w = 0;
    }

    RLdata[n] = assign_c;
    RWdata[n] = assign_w;
    // RWdata[n] = 1;
  }

  cur_iter_++;
  if (cur_iter_ % display_ == 0) {
    printf(
        "RoILabel %d\tfg_rois: %d\tbg_rois: %d\tfg_weight: %f\tbg_weight: %f\n",
        uuid_, acc_fg_rois_ / display_, acc_bg_rois_ / display_,
        acc_fg_weight_ / acc_fg_rois_, acc_bg_weight_ / acc_bg_rois_);
    acc_fg_rois_ = 0;
    acc_bg_rois_ = 0;
    acc_fg_weight_ = 0;
    acc_bg_weight_ = 0;
  }

  return true;
}

REGISTER_CPU_OPERATOR(RoILabel, RoILabelOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(RoILabel)
    .NumInputs(3, 4)
    .NumOutputs(2)
    .SetDoc(R"DOC(
)DOC")
    .Arg("debug_info", "(bool) default to false")
    .Input(0, "S", "input Score tensor of size (n x c(+1))")
    .Input(1, "U", "input IoU tensor of size (n x n)")
    .Input(2, "L", "input Label tensor of size (1 x c)")
    .Output(0, "RL", "output RoI Label tensor of size (n)")
    .Output(1, "RW", "output RoI Label tensor of size (n x (c+1))");

namespace {

NO_GRADIENT(RoILabel);

}  // namespace

}  // namespace caffe2
