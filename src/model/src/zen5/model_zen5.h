#ifndef EINSUM_IR_MODEL_ZEN5_MODEL_ZEN5_H
#define EINSUM_IR_MODEL_ZEN5_MODEL_ZEN5_H

#include <algorithm>

#include "bench_zen5.h"

namespace einsum_ir::model::common {
  enum class DType;
}

namespace einsum_ir::model::zen5 {

  /**
   * Find surrounding indices and interpolation factor for M dimension.
   * Handles special modulo-16 mapping for values above 128.
   *
   * @param arr The array of dimension values.
   * @param size The size of the dimension array.
   * @param val The target dimension value.
   * @param idx_lower Output parameter for the lower index.
   * @param t Output parameter for the interpolation factor.
   */
  void find_bounds_m(const int* arr, int size, int val, int& idx_lower, double& t);

  /**
   * Get interpolated GFLOPS value based on input dimensions and transpose flags.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_trans_a The transpose flag for matrix A (0 or 1).
   * @param i_trans_b The transpose flag for matrix B (0 or 1).
   * @param i_dtype The data type (FP32 or FP64).
   *
   * @return The interpolated GFLOPS value.
   */
  double get_interpolated_gflops(int i_m,
                                 int i_n,
                                 int i_k,
                                 int i_trans_a,
                                 int i_trans_b,
                                 einsum_ir::model::common::DType i_dtype);

}  // namespace einsum_ir::model::zen5

#endif  // EINSUM_IR_MODEL_ZEN5_MODEL_ZEN5_H