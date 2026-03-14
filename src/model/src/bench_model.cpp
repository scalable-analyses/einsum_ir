#include <cstring>
#include <iostream>

#include "libxsmm.h"
#include "common/common.h"

double get_time_xsmm(int i_m,
                      int i_n,
                      int i_k,
                      int i_trans_a,
                      int i_trans_b,
                      double& o_gflops) {

  if (i_m <= 0 || i_n <= 0 || i_k <= 0) {
    std::cerr << "Matrix dimensions must be positive" << std::endl;
    return 0.0;
  }

  if (i_trans_a < 0 || i_trans_a > 1 || i_trans_b < 0 || i_trans_b > 1) {
    std::cerr << "Transpose flags must be 0 or 1" << std::endl;
    return 0.0;
  }

  float* l_a;
  float* l_b;
  float* l_c;
  float* l_c_ref;

  char l_trans_a = (i_trans_a == 0) ? 'N' : 'T';
  char l_trans_b = (i_trans_b == 0) ? 'N' : 'T';

  posix_memalign((void**)&l_a, 128, i_m * i_k * sizeof(float));
  posix_memalign((void**)&l_b, 128, i_k * i_n * sizeof(float));
  posix_memalign((void**)&l_c, 128, i_m * i_n * sizeof(float));
  posix_memalign((void**)&l_c_ref, 128, i_m * i_n * sizeof(float));

  libxsmm_gemm_shape l_shape_gemm;
  libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS(l_trans_a, l_trans_b);
  libxsmm_bitfield l_prefetch_flags_brgemm = 0;

  l_shape_gemm = libxsmm_create_gemm_shape(i_m,
                                            i_n,
                                            i_k,
                                            (i_trans_a == 0) ? i_m : i_k,
                                            (i_trans_b == 0) ? i_k : i_n,
                                            i_m,
                                            libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                            libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                            libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                            libxsmm_datatype::LIBXSMM_DATATYPE_F32);

  libxsmm_gemm_batch_reduce_config l_config;
  l_config.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
  l_config.br_stride_a_hint = 0;
  l_config.br_stride_b_hint = 0;
  l_config.br_unroll_hint = 0;

  libxsmm_xmmfunction l_xmm_gemm_beta_1;
  l_xmm_gemm_beta_1.gemm = libxsmm_dispatch_brgemm(l_shape_gemm,
                                                    l_flags_brgemm,
                                                    l_prefetch_flags_brgemm,
                                                    l_config);

  std::random_device l_rd;
  std::mt19937 l_gen(l_rd());
  std::normal_distribution<float> l_dist(0.0, 1.0);

  for (int64_t l_en = 0; l_en < i_m * i_k; l_en++) {
    l_a[l_en] = l_dist(l_gen);
  }

  for (int64_t l_en = 0; l_en < i_k * i_n; l_en++) {
    l_b[l_en] = l_dist(l_gen);
  }

  for (int64_t l_en = 0; l_en < i_m * i_n; l_en++) {
    l_c[l_en] = 0.0f;
    l_c_ref[l_en] = 0.0f;
  }

  size_t l_reps_warmup = 10;
  auto l_start_warmup = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < l_reps_warmup; ++i) {
    libxsmm_gemm_param l_param;
    l_param.a.primary = l_a;
    l_param.b.primary = l_b;
    l_param.c.primary = l_c;
    l_xmm_gemm_beta_1.gemm(&l_param);
  }
  auto l_end_warmup = std::chrono::high_resolution_clock::now();

  double l_warmup_duration = std::chrono::duration<double>(l_end_warmup - l_start_warmup).count();
  double time_per_iter = l_warmup_duration / l_reps_warmup;
  size_t l_reps = (size_t)(8.0 / time_per_iter);
  if (l_reps < 1) l_reps = 1;

  auto l_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < l_reps; ++i) {
    libxsmm_gemm_param l_param;
    l_param.a.primary = l_a;
    l_param.b.primary = l_b;
    l_param.c.primary = l_c;
    l_xmm_gemm_beta_1.gemm(&l_param);
  }
  auto l_end = std::chrono::high_resolution_clock::now();

  double l_duration = std::chrono::duration<double>(l_end - l_start).count();

  double gflops = (2.0 * i_m * i_n * i_k * l_reps) / (l_duration * 1.0e9);

  o_gflops = gflops;

  double time = l_duration / l_reps;

  free(l_a);
  free(l_b);
  free(l_c);
  free(l_c_ref);

  return time;
}

int main(int argc, char** argv) {
  if (argc < 7 || argc > 9) {
    std::cout << "Usage: " << argv[0] << " <m> <n> <k> <trans_a> <trans_b> <model> [peak_gflops] [vector_size]" << std::endl;
    std::cout << "  m, n, k:       Matrix dimensions (positive integers)" << std::endl;
    std::cout << "  trans_a:       Transpose A matrix (0 or 1)" << std::endl;
    std::cout << "  trans_b:       Transpose B matrix (0 or 1)" << std::endl;
    std::cout << "  model:         Performance model to use (zen5, m4, a76, generic)" << std::endl;
    std::cout << "  peak_gflops:   [Optional, generic only] Peak GFLOPS of architecture" << std::endl;
    std::cout << "  vector_size:   [Optional, generic only] Vector width in byte" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << argv[0] << " 64 48 64 0 0 zen5" << std::endl;
    std::cout << "  " << argv[0] << " 64 48 64 0 0 generic 100.0 16" << std::endl;
    return EXIT_FAILURE;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);
  int trans_a = std::atoi(argv[4]);
  int trans_b = std::atoi(argv[5]);
  einsum_ir::model::common::Model model;

  double peak_gflops = 0.0;
  int vector_size = 0;

  if (std::strcmp(argv[6], "zen5") == 0) {
    model = einsum_ir::model::common::Model::ZEN5;
  } else if (std::strcmp(argv[6], "m4") == 0) {
    model = einsum_ir::model::common::Model::M4;
  } else if (std::strcmp(argv[6], "a76") == 0) {
    model = einsum_ir::model::common::Model::A76;
  } else if (std::strcmp(argv[6], "generic") == 0) {
    model = einsum_ir::model::common::Model::GENERIC;

    if (argc >= 8) {
      peak_gflops = std::atof(argv[7]);
    }
    if (argc >= 9) {
      vector_size = std::atoi(argv[8]);
    }

    if (peak_gflops <= 0.0 || vector_size <= 0) {
      std::cerr << "Error: For generic model, you must provide peak_gflops > 0 and vector_size > 0" << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    std::cerr << "Error: Unknown model '" << argv[6] << "'" << std::endl;
    std::cerr << "Available models: zen5, m4, a76, generic" << std::endl;
    return EXIT_FAILURE;
  }

  if (model != einsum_ir::model::common::Model::GENERIC && argc > 7) {
    std::cerr << "Warning: Extra parameters ignored for non-generic models" << std::endl;
  }

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Matrix Config:" << std::endl;
  std::cout << "M: " << m << ", N: " << n << ", K: " << k << ", TransA: " << trans_a << ", TransB: " << trans_b << std::endl;
  std::cout << "Model: " << (model == einsum_ir::model::common::Model::ZEN5 ? "zen5" : model == einsum_ir::model::common::Model::M4 ? "m4"
                                                                                   : model == einsum_ir::model::common::Model::A76  ? "a76"
                                                                                                                                    : "generic")
            << std::endl;

  if (model == einsum_ir::model::common::Model::GENERIC) {
    std::cout << "Peak GFLOPS: " << peak_gflops << std::endl;
    std::cout << "Vector Size: " << vector_size << std::endl;
  }

  double model_gflops = 0.0;
  double xsmm_gflops = 0.0;

  double model_time = einsum_ir::model::common::get_time_model(m, n, k, trans_a, trans_b,einsum_ir::model::common::DType::FP32, model, model_gflops, peak_gflops, vector_size);
  double xsmm_time = get_time_xsmm(m, n, k, trans_a, trans_b, xsmm_gflops);

  std::cout << "Model GFLOPS: " << model_gflops << std::endl;
  std::cout << "XSMM GFLOPS: " << xsmm_gflops << std::endl;
  std::cout << "Model Time: " << model_time << " seconds" << std::endl;
  std::cout << "XSMM Time: " << xsmm_time << " seconds" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  return EXIT_SUCCESS;
}