#include "common.h"

#include <stdexcept>

namespace einsum_ir::model::common {

  double get_time_model(int i_m,
                        int i_n,
                        int i_k,
                        int i_trans_a,
                        int i_trans_b,
                        DType i_dtype,
                        Model i_model,
                        double& o_gflops,
                        double i_peak_gflops,
                        int i_vector_size) {
    if (i_m <= 0 || i_n <= 0 || i_k <= 0) {
      std::cerr << "Matrix dimensions must be positive" << std::endl;
      return 0.0;
    }

    if (i_trans_a < 0 || i_trans_a > 1 || i_trans_b < 0 || i_trans_b > 1) {
      std::cerr << "Transpose flags must be 0 or 1" << std::endl;
      return 0.0;
    }

    if (i_model == Model::GENERIC) {
      if (i_peak_gflops <= 0.0) {
        std::cerr << "Peak GFLOPS must be positive for generic model" << std::endl;
        return 0.0;
      }
      if (i_vector_size <= 0) {
        std::cerr << "Vector size must be positive for generic model" << std::endl;
        return 0.0;
      }
    }

    double gflops = 1.0;
    switch (i_model) {
      case Model::ZEN5:
        gflops = einsum_ir::model::zen5::get_interpolated_gflops(i_m, i_n, i_k, i_trans_a, i_trans_b, i_dtype);
        break;
      case Model::M4:
        gflops = einsum_ir::model::m4::get_interpolated_gflops(i_m, i_n, i_k, i_trans_b, i_dtype);
        break;
      case Model::A76:
        gflops = einsum_ir::model::a76::get_interpolated_gflops(i_m, i_n, i_k, i_trans_a, i_trans_b, i_dtype);
        break;
      case Model::GENERIC:
        gflops = einsum_ir::model::generic::get_gflops(i_m, i_n, i_k, i_trans_a, i_trans_b, i_dtype, i_peak_gflops, i_vector_size);
        break;
    }
    o_gflops = gflops;
    double time = ((double)(i_m) * (double)(i_n) * (double)(i_k) * 2.0) / (gflops * 1.0e9);
    return time;
  }

}  // namespace einsum_ir::model::common
