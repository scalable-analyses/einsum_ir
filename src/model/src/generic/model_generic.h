#ifndef EINSUM_IR_MODEL_GENERIC_MODEL_GENERIC_H
#define EINSUM_IR_MODEL_GENERIC_MODEL_GENERIC_H

namespace einsum_ir::model::common {
  enum class DType;
}

namespace einsum_ir::model::generic {
  /**
   * Get GFLOPS value based on input dimensions.
   * Using a heuristic model.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_trans_a The transpose flag for matrix A (0 or 1).
   * @param i_trans_b The transpose flag for matrix B (0 or 1).
   * @param i_dtype The data type (FP32 or FP64).
   * @param i_peak_gflops The peak GFLOPS of the architecture.
   * @param i_vector_size The vector width in bytes.
   *
   * @return The interpolated GFLOPS value.
   */
  double get_gflops(int i_m,
                    int i_n,
                    int i_k,
                    int i_trans_a,
                    int i_trans_b,
                    einsum_ir::model::common::DType i_dtype,
                    double i_peak_gflops,
                    int i_vector_size);
}  // namespace einsum_ir::model::generic

#endif  // EINSUM_IR_MODEL_GENERIC_MODEL_GENERIC_H