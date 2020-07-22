#ifndef CAFFE2_OPERATORS_DenseCRF_OP_H_
#define CAFFE2_OPERATORS_DenseCRF_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"
#include "densecrf.h"

namespace caffe2 {

template <typename T, class Context>
class DenseCRFOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  DenseCRFOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        max_iter_(this->template GetSingleArgument<int>("max_iter", 10)),
        scale_factor_(this->template GetSingleArgument<int>("scale_factor", 1)),
        color_factor_(
            this->template GetSingleArgument<int>("color_factor", 13)),
        SIZE_STD(this->template GetSingleArgument<float>("SIZE_STD", 500)),
        POS_W(this->template GetSingleArgument<float>("POS_W", 3)),
        POS_X_STD(this->template GetSingleArgument<float>("POS_X_STD", 3)),
        POS_Y_STD(this->template GetSingleArgument<float>("POS_Y_STD", 3)),
        BI_W(this->template GetSingleArgument<float>("BI_W", 10)),
        BI_X_STD(this->template GetSingleArgument<float>("BI_X_STD", 80)),
        BI_Y_STD(this->template GetSingleArgument<float>("BI_Y_STD", 80)),
        BI_R_STD(this->template GetSingleArgument<float>("BI_R_STD", 13)),
        BI_G_STD(this->template GetSingleArgument<float>("BI_G_STD", 13)),
        BI_B_STD(this->template GetSingleArgument<float>("BI_B_STD", 13)),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)) {}
  ~DenseCRFOp() {}

  bool RunOnDevice() override;

  void set_unary_energy(const float* unary_costs_ptr);

  void add_pairwise_energy(float w1, float theta_alpha_1, float theta_alpha_2,
                           float theta_betta_1, float theta_betta_2,
                           float theta_betta_3, float w2, float theta_gamma_1,
                           float theta_gamma_2, const unsigned char* im);

  void map(int n_iters, int* result);
  void inference(int n_iters, float* result);

  int npixels();
  int nlabels();

  void dense_crf(const unsigned char* image, const float* unary,
                 float* probs_out);

 protected:
  int max_iter_;
  int scale_factor_;
  int color_factor_;

  float SIZE_STD;

  float POS_W;
  float POS_X_STD;
  float POS_Y_STD;
  float BI_W;
  float BI_X_STD;
  float BI_Y_STD;
  float BI_R_STD;
  float BI_G_STD;
  float BI_B_STD;

  bool debug_info_;

  DenseCRF2D* m_crf;

  int H;
  int W;
  int m_nlabels;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_DenseCRF_OP_H_
