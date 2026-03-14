#include "model_generic.h"

#include <iostream>

#include "../common/common.h"

namespace einsum_ir::model::generic {

  double get_gflops(int i_m,
                    int i_n,
                    int i_k,
                    int i_trans_a,
                    int i_trans_b,
                    einsum_ir::model::common::DType i_dtype,
                    double i_peak_gflops,
                    int i_vector_size) {
    (void)i_dtype;
    double m_factor = 1.0f;
    double n_factor = 1.0f;
    double k_factor = 1.0f;
    double efficiency = 1.0f;
    double gflops = 0.0f;
    if (i_vector_size > 0) {
      int kernel_m_size = i_vector_size;
      int num_kernels = i_m / kernel_m_size;
      int remainder = i_m % kernel_m_size;

      double base_penalty = 0.5 * (1.0 - static_cast<double>(remainder) / static_cast<double>(kernel_m_size));

      double penalty_reduction = 1.0;
      if (num_kernels > 0) {
        penalty_reduction = 1.0 / (1.0 + num_kernels);
      }

      m_factor = 1.0 - (base_penalty * penalty_reduction);
    }

    if (i_k >= 48) {
      k_factor = 1.0f;
    } else {
      k_factor = 0.7f + (0.3f * static_cast<double>(i_k - 1) / 47.0f);
    }

    if (i_n >= 8) {
      n_factor = 1.0f;
    } else {
      n_factor = 0.7f + (0.3f * static_cast<double>(i_n - 1) / 8.0f);
    }

    efficiency = static_cast<double>(m_factor * n_factor * k_factor);

    gflops = i_peak_gflops * efficiency;

    if (i_trans_a != 0) {
      gflops *= 0.9;
    }

    return gflops;
  }
}  // namespace einsum_ir::model::generic