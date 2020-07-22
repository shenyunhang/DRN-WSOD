#include <math.h>
#include <Eigen/Core>
#include <algorithm>
#include <functional>

#include "crf_op.h"

namespace caffe2 {

namespace {

template <typename T>
void bilinear_interpolation(const float* input, float* output,
                            const int batch_size, const int num_channels,
                            const int input_height, const int input_width,
                            const int output_height, const int output_width) {
  int channels = num_channels * batch_size;

  const float rheight = (output_height > 1)
                            ? (float)(input_height - 1) / (output_height - 1)
                            : 0.f;
  const float rwidth =
      (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
  for (int h2 = 0; h2 < output_height; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < input_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = (float)1. - h1lambda;
    for (int w2 = 0; w2 < output_width; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < input_width - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = (float)1. - w1lambda;
      const float* Xdata = &input[h1 * input_width + w1];
      float* Ydata = &output[h2 * output_width + w2];
      for (int c = 0; c < channels; ++c) {
        Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) +
                   h1lambda * (w0lambda * Xdata[h1p * input_width] +
                               w1lambda * Xdata[h1p * input_width + w1p]);
        Xdata += input_width * input_height;
        Ydata += output_width * output_height;
      }
    }
  }
}

template <typename T>
void image_process(const float* input, unsigned char* output,
                   const int batch_size, const int height, const int width) {
  // TODO(YH): add argument
  float mean[] = {102.9801, 115.9465, 122.7717};
  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int idx_i = ((b * 3 + c) * height + h) * width + w;
          int idx_o = ((b * height + h) * width + w) * 3 + c;
          output[idx_o] = (unsigned char)(input[idx_i] + mean[c]);
        }
      }
    }
  }
}

template <typename T>
void unary_process(const float* input, float* output, const int batch_size,
                   const int num_classes, const int height, const int width) {
  const float min_prob = 0.0001;
  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < num_classes; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int idx_i = ((b * num_classes + c) * height + h) * width + w;
          int idx_o = ((b * height + h) * width + w) * num_classes + c;
          output[idx_o] = std::max(input[idx_i], min_prob);
          // output[idx_o] = -1. * std::max(input[idx_i], min_prob);
          // output[idx_o] = -1. * input[idx_i];
        }
      }
    }
  }
}

template <typename T>
void result_process(const float* input, float* output, const int batch_size,
                    const int num_classes, const int height, const int width) {
  const float min_prob = 0.0001;

  Tensor N(caffe2::CPU);
  N.Resize(batch_size, height, width);
  float* Nmdata = N.mutable_data<float>();

  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        Nmdata[0] = 0;
        for (int c = 0; c < num_classes; c++) {
          int idx_i = ((b * height + h) * width + w) * num_classes + c;
          int idx_o = ((b * num_classes + c) * height + h) * width + w;
          output[idx_o] = std::max(input[idx_i], min_prob);
          Nmdata[0] += output[idx_o];
        }
        Nmdata += 1;
      }
    }
  }

  const float* Ndata = N.data<float>();

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < num_classes; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int idx_o = ((b * num_classes + c) * height + h) * width + w;
          float norm = *(Ndata + (b * height + h) * width + w);
          // output[idx_o] = log(output[idx_o] / norm);
          output[idx_o] = output[idx_o] / norm;
        }
      }
    }
  }
}

}  // namespace

template <>
int DenseCRFOp<float, CPUContext>::npixels() {
  return W * H;
}

template <>
int DenseCRFOp<float, CPUContext>::nlabels() {
  return m_nlabels;
}

template <>
void DenseCRFOp<float, CPUContext>::add_pairwise_energy(
    float w1, float theta_alpha_1, float theta_alpha_2, float theta_betta_1,
    float theta_betta_2, float theta_betta_3, float w2, float theta_gamma_1,
    float theta_gamma_2, const unsigned char* im) {
  m_crf->addPairwiseGaussian(theta_gamma_1, theta_gamma_2,
                             new PottsCompatibility(w2));
  m_crf->addPairwiseBilateral(theta_alpha_1, theta_alpha_2, theta_betta_1,
                              theta_betta_2, theta_betta_3, im,
                              new PottsCompatibility(w1));
  // m_crf->addPairwiseGaussian(3, 3, new PottsCompatibility(3));
  // m_crf->addPairwiseBilateral(80, 80, 13, 13, 13, im,
  // new PottsCompatibility(10));
}

template <>
void DenseCRFOp<float, CPUContext>::set_unary_energy(
    const float* unary_costs_ptr) {
  m_crf->setUnaryEnergy(
      Eigen::Map<const Eigen::MatrixXf>(unary_costs_ptr, m_nlabels, W * H));
}

template <>
void DenseCRFOp<float, CPUContext>::map(int n_iters, int* labels) {
  VectorXs labels_vec = m_crf->map(n_iters);
  for (int i = 0; i < (W * H); ++i) labels[i] = labels_vec(i);
}

template <>
void DenseCRFOp<float, CPUContext>::inference(int n_iters, float* probs_out) {
  MatrixXf probs = m_crf->inference(n_iters);
  for (int i = 0; i < npixels(); ++i)
    for (int j = 0; j < nlabels(); ++j)
      probs_out[i * nlabels() + j] = probs(j, i);
}

template <>
void DenseCRFOp<float, CPUContext>::dense_crf(const unsigned char* image,
                                              const float* unary,
                                              float* probs_out) {
  // set unary potentials
  set_unary_energy(unary);

  // set pairwise potentials
  // add_pairwise_energy(10, 80 / scale_factor_, 80 / scale_factor_,
  // color_factor_, color_factor_, color_factor_, 3, 3 / scale_factor_, 3 /
  // scale_factor_, image);
  add_pairwise_energy(BI_W, BI_X_STD / scale_factor_, BI_Y_STD / scale_factor_,
                      BI_R_STD, BI_G_STD, BI_B_STD, POS_W,
                      POS_X_STD / scale_factor_, POS_Y_STD / scale_factor_,
                      image);

  // run inference
  inference(max_iter_, probs_out);
}

template <>
bool DenseCRFOp<float, CPUContext>::RunOnDevice() {
  const auto& U = Input(0);
  const auto& I = Input(1);

  CAFFE_ENFORCE_EQ(U.dim(), 4);
  CAFFE_ENFORCE_EQ(I.dim(), 4);
  CAFFE_ENFORCE_EQ(U.dim32(0), I.dim32(0));
  CAFFE_ENFORCE_EQ(I.dim32(1), 3);

  const int batch_size = U.dim32(0);
  const int num_classes = U.dim32(1);
  const int height = U.dim32(2);
  const int width = U.dim32(3);
  const int height_im = I.dim32(2);
  const int width_im = I.dim32(3);
  H = height;
  W = width;
  m_nlabels = num_classes;

  m_crf = new DenseCRF2D(W, H, m_nlabels);

  auto* M = Output(0);
  M->Resize(batch_size, num_classes, height, width);

  Tensor MT(caffe2::CPU);
  MT.Resize(batch_size, height, width, num_classes);

  Tensor IT(caffe2::CPU);
  IT.Resize(batch_size, height, width, 3);

  if (height != height_im || width != width_im) {
    Tensor IB(caffe2::CPU);
    IB.Resize(batch_size, 3, height, width);
    bilinear_interpolation<float>(I.data<float>(), IB.mutable_data<float>(),
                                  batch_size, 3, height_im, width_im, height,
                                  width);

    image_process<float>(IB.data<float>(), IT.mutable_data<unsigned char>(),
                         batch_size, height, width);
  } else {
    image_process<float>(I.data<float>(), IT.mutable_data<unsigned char>(),
                         batch_size, height, width);
  }

  Tensor UT(caffe2::CPU);
  UT.Resize(batch_size, height, width, num_classes);
  unary_process<float>(U.data<float>(), UT.mutable_data<float>(), batch_size,
                       num_classes, height, width);

  for (int b = 0; b < batch_size; b++) {
    const unsigned char* image =
        IT.data<unsigned char>() + b * height * width * 3;
    const float* unary = UT.data<float>() + b * height * width * num_classes;
    float* probs_out =
        MT.mutable_data<float>() + b * height * width * num_classes;

    // auto adjust scale_factor_
    scale_factor_ = 1.0 * SIZE_STD / std::max(height, width);

    dense_crf(image, unary, probs_out);
  }

  result_process<float>(MT.data<float>(), M->mutable_data<float>(), batch_size,
                        num_classes, height, width);

  delete m_crf;
  return true;
}

REGISTER_CPU_OPERATOR(DenseCRF, DenseCRFOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(DenseCRF)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
)DOC")
    .Arg("max_iter", "(int) default to 0")
    .Arg("debug_info", "(bool) default to false")
    .Input(0, "U", "input tensor of size (BxCxH1xW1)")
    .Input(1, "I", "input tensor of size (Bx3xH2xW2)")
    .Output(0, "M", "output tensor of size (BxCxH1xW1)");

namespace {

NO_GRADIENT(DenseCRF);

}  // namespace

}  // namespace caffe2
