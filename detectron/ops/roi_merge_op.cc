#include <functional>

#include <cfloat>
#include <cmath>
#include "roi_merge_op.h"

namespace caffe2 {

float getlambda(float iter, float max_iter) {
  float low_bound = 0.01;

  float lambda = (log(iter + low_bound) - log(low_bound)) /
                 (log(max_iter + low_bound) - log(low_bound));
  return lambda;
}

template <typename T>
vector<size_t> sort_indexes(const vector<T>& v) {
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

template <>
bool RoIMergeOp<float, CPUContext>::RunOnDevice() {
  const auto& S = Input(0);
  const auto& J = Input(1);
  const auto& C = Input(2);
  const auto& D = Input(3);

  CAFFE_ENFORCE_EQ(S.dim(), 2);
  CAFFE_ENFORCE_EQ(J.dim(), 2);
  CAFFE_ENFORCE_EQ(C.dim(), 2);
  CAFFE_ENFORCE_EQ(D.dim(), 2);
  CAFFE_ENFORCE_EQ(S.dim32(0), J.dim32(0));
  CAFFE_ENFORCE_EQ(S.dim32(0), C.dim32(0));
  CAFFE_ENFORCE_EQ(S.dim32(0), D.dim32(0));
  CAFFE_ENFORCE_EQ(S.dim32(1), 1);
  CAFFE_ENFORCE_EQ(J.dim32(0), J.dim32(1));
  CAFFE_ENFORCE_EQ(C.dim32(1), D.dim32(1));

  const int num_roi = C.dim32(0);
  const int num_class = C.dim32(1);

  auto* MC = Output(0);
  auto* MD = Output(1);
  auto* I = Output(2);
  auto* IC = Output(3);

  I->Resize(num_roi);
  math::Set<int, CPUContext>(I->numel(), -1, I->template mutable_data<int>(),
                             &context_);

  const float* Sdata = S.data<float>();
  const float* Jdata = J.data<float>();
  const float* Cdata = C.data<float>();
  const float* Ddata = D.data<float>();

  int* Idata = I->template mutable_data<int>();

  // sort score
  std::set<int> rois_idx;

  std::vector<float> SSdata;
  SSdata.clear();

  for (int n = 0; n < num_roi; n++) {
    SSdata.push_back(Sdata[n]);
  }
  vector<size_t> sort_idx = sort_indexes(SSdata);

  float lambda =
      getlambda(float(cur_iter_) / float(size_epoch_), float(max_epoch_));
  int cur_id = 0;
  int top_k = num_roi > 200 ? 200 : num_roi;

  // merge top
  for (int t = 0; t < top_k; t++) {
    int n = sort_idx[t];
    if (Idata[n] == -1) {
    } else {
      continue;
    }
    Idata[n] = cur_id;

    int end_num = t + 40 > top_k ? top_k : t + 40;

    for (int tt = t; tt < end_num; tt++) {
      int i = sort_idx[tt];
      if (Idata[i] == -1) {
      } else {
        continue;
      }

      bool flag_in_clique = true;

      for (int ttt = t; ttt < end_num; ttt++) {
        int j = sort_idx[ttt];
        if (Idata[j] == cur_id) {
        } else {
          continue;
        }

        if (Jdata[i * num_roi + j] < lambda) {
          flag_in_clique = false;
          break;
        }
      }
      if (flag_in_clique) {
        Idata[i] = cur_id;
      }
    }
    cur_id += 1;
  }

  // for display
  int num_top_id = cur_id;

  // merge rest
  for (int n = 0; n < num_roi; n++) {
    if (Idata[n] == -1) {
    } else {
      continue;
    }

    Idata[n] = cur_id;
    cur_id += 1;
  }

  int num_id = cur_id;

  MC->Resize(num_id, num_class);
  MD->Resize(num_id, num_class);
  math::Set<float, CPUContext>(MC->numel(), 0.f,
                               MC->template mutable_data<float>(), &context_);
  math::Set<float, CPUContext>(MD->numel(), 0.f,
                               MD->template mutable_data<float>(), &context_);

  IC->Resize(num_id);
  math::Set<int, CPUContext>(IC->numel(), 0, IC->template mutable_data<int>(),
                             &context_);

  float* MCdata = MC->template mutable_data<float>();
  float* MDdata = MD->template mutable_data<float>();
  int* ICdata = IC->template mutable_data<int>();

  // count ID
  for (int n = 0; n < num_roi; n++) {
    int id = Idata[n];
    ICdata[id] += 1;
  }

  // for display
  int max_clique = 0;
  int min_clique = top_k;
  for (int i = 0; i < num_top_id; i++) {
    if (ICdata[i] > max_clique) {
      max_clique = ICdata[i];
    }
    if (ICdata[i] < min_clique) {
      min_clique = ICdata[i];
    }
  }
  acc_num_top_id_ += num_top_id;
  acc_max_clique_ += max_clique;
  acc_min_clique_ += min_clique;

  // merge score
  for (int n = 0; n < num_roi; n++) {
    int id = Idata[n];
    for (int c = 0; c < num_class; c++) {
      MCdata[id * num_class + c] += Cdata[n * num_class + c] / ICdata[id];
      MDdata[id * num_class + c] += Ddata[n * num_class + c] / ICdata[id];
    }
  }

  cur_iter_++;

  if (cur_iter_ % display_ == 0) {
    printf(
        "RoIMerge %d\tlambda: %f\tacc_top_num_id_: %d\tacc_max_clique: "
        "%d\tacc_min_clique: %d\n",
        cur_iter_, lambda, acc_num_top_id_ / display_,
        acc_max_clique_ / display_, acc_min_clique_ / display_);

    acc_num_top_id_ = 0;
    acc_max_clique_ = 0;
    acc_min_clique_ = 0;
  }

  return true;
}  // namespace caffe2

template <>
bool RoIMergeGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& C = Input(0);
  const auto& D = Input(1);
  const auto& GMC = Input(2);
  const auto& GMD = Input(3);
  const auto& I = Input(4);
  const auto& IC = Input(5);

  const int num_roi = C.dim32(0);
  const int num_class = C.dim32(1);

  const float* GMCdata = GMC.data<float>();
  const float* GMDdata = GMD.data<float>();
  const int* Idata = I.data<int>();
  const int* ICdata = IC.data<int>();

  auto* GC = Output(0);
  auto* GD = Output(1);

  GC->Resize(num_roi, num_class);
  GD->Resize(num_roi, num_class);

  float* GCdata = GC->template mutable_data<float>();
  float* GDdata = GD->template mutable_data<float>();

  for (int n = 0; n < num_roi; n++) {
    int id = Idata[n];
    for (int c = 0; c < num_class; c++) {
      GCdata[n * num_class + c] = GMCdata[id * num_class + c] / ICdata[id];
      GDdata[n * num_class + c] = GMDdata[id * num_class + c] / ICdata[id];
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(RoIMerge, RoIMergeOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(RoIMergeGradient, RoIMergeGradientOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(RoIMerge)
    .NumInputs(4)
    .NumOutputs(4)
    .SetDoc(R"DOC(
)DOC")
    .Arg("debug_info", "(bool) default to false")
    .Input(0, "S", "input Score tensor of size (n x c))")
    .Input(1, "J", "input IoU tensor of size (n x n)")
    .Input(2, "C", "input Score tensor of size (n x c)")
    .Input(3, "D", "input Score tensor of size (n x c)")
    .Output(0, "MC", "output Score tensor of size (n x c)")
    .Output(1, "MD", "output Score tensor of size (n x c)")
    .Output(2, "I", "output Index tensor of size (n)")
    .Output(3, "IC", "output Index Count tensor of size (m)");

OPERATOR_SCHEMA(RoIMergeGradient).NumInputs(6).NumOutputs(2);

namespace {

class GetRoIMergeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    if (GradOut(0).IsEmpty()) {
      return {};
    }
    vector<string> ins;
    ins.push_back(I(2));
    ins.push_back(I(3));
    ins.push_back(GO(0));
    ins.push_back(GO(1));
    ins.push_back(O(2));
    ins.push_back(O(3));

    vector<string> outs;
    outs.push_back(GI(2));
    outs.push_back(GI(3));

    return SingleGradientDef("RoIMergeGradient", "", ins, outs);
  }
};

REGISTER_GRADIENT(RoIMerge, GetRoIMergeGradient);

}  // namespace

}  // namespace caffe2
