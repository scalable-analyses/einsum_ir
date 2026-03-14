#ifndef EINSUM_IR_MODEL_A76_MODEL_A76_H
#define EINSUM_IR_MODEL_A76_MODEL_A76_H

#include <algorithm>
#include <iostream>

#include "bench_a76.h"

namespace einsum_ir::model::common {
  enum class DType;
}

namespace einsum_ir::model::a76 {

  /**
   * Microkernel size configuration for M and N dimensions.
   */
  struct microkernel_sizes {
    unsigned int m;
    unsigned int n;
  };

  /**
   * JIT kernel blocking configuration.
   * Defines a 2x2 grid of microkernels with varying sizes:
   * -----------
   * | K1 | K3 |
   * -----------
   * | K2 | K4 |
   * -----------
   */
  struct jit_sizes {
    microkernel_sizes k1;  //!< Full M, first N block
    microkernel_sizes k2;  //!< Rest M, first N block
    microkernel_sizes k3;  //!< Full M, second N block
    microkernel_sizes k4;  //!< Rest M, second N block

    unsigned int m_count_full_count;  //!< Number of full M blocks

    unsigned int first_n_count;   //!< Count of first N size blocks
    unsigned int second_n_count;  //!< Count of second N size blocks
  };

  /**
   * Find exact index in a sorted array or return closest/clamped value.
   * Used for M and N dimension lookups where exact matches are expected.
   *
   * @param arr The sorted array to search.
   * @param size The size of the array.
   * @param val The value to find.
   * @return The index of the exact match, or clamped/closest index.
   */
  int get_exact_index(const int* arr, int size, int val);

  /**
   * Get GFLOPS performance for a specific kernel size with K interpolation.
   * Uses the gflops_table with linear interpolation for K dimension only.
   * M and N are looked up exactly.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_trans_a The transpose flag for matrix A (0 or 1).
   * @param i_trans_b The transpose flag for matrix B (0 or 1).
   * @return The interpolated GFLOPS value for this kernel configuration.
   */
  double get_gflops(int i_m, int i_n, int i_k, int i_trans_a, int i_trans_b);

  /**
   * Determines microkernel sizes based on register constraints.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_transpose_a The transpose flag for matrix A (0 or 1).
   * @param i_transpose_b The transpose flag for matrix B (0 or 1).
   * @param kernels Output parameter for the calculated blocking configuration.
   */
  void get_blocking(int64_t i_m,
                    int64_t i_n,
                    int64_t i_k,
                    int i_transpose_a,
                    int i_transpose_b,
                    jit_sizes& kernels);

  /**
   * Calculate aggregate GFLOPS for a GEMM operation using blocking strategy.
   * Computes weighted average performance across all microkernel blocks.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param kernels The blocking configuration.
   * @param i_transpose_a The transpose flag for matrix A (0 or 1).
   * @param i_transpose_b The transpose flag for matrix B (0 or 1).
   * @return The aggregate GFLOPS performance estimate.
   */
  double calculate_gflops(int i_m,
                          int i_n,
                          int i_k,
                          jit_sizes& kernels,
                          int i_transpose_a,
                          int i_transpose_b);

  /**
   * Get interpolated GFLOPS estimate for a GEMM operation.
   * High-level function that automatically computes blocking and performance.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_transpose_a The transpose flag for matrix A (0 or 1).
   * @param i_transpose_b The transpose flag for matrix B (0 or 1).
   * @param i_dtype The data type (FP32 or FP64).
   * @return The interpolated GFLOPS value.
   */
  double get_interpolated_gflops(int i_m,
                                 int i_n,
                                 int i_k,
                                 int i_transpose_a,
                                 int i_transpose_b,
                                 einsum_ir::model::common::DType i_dtype);

}  // namespace einsum_ir::model::a76

#endif  // EINSUM_IR_MODEL_A76_MODEL_A76_H
