#ifndef EINSUM_IR_MODEL_A76_BENCH_A76_H
#define EINSUM_IR_MODEL_A76_BENCH_A76_H

namespace einsum_ir::model::a76 {

  //! Dimension sizes for the ARM Cortex-A76 performance lookup table
  static const int M_SIZE = 16;
  static const int N_SIZE = 15;
  static const int K_SIZE = 7;

  //! Tested M dimension values (microkernel sizes)
  static const int M_VALUES[M_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  //! Tested N dimension values (microkernel sizes)
  static const int N_VALUES[N_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  //! Tested K dimension values (contracting dimension sizes)
  static const int K_VALUES[K_SIZE] = {1, 4, 16, 24, 32, 64, 128};

  //! GFLOPS lookup table indexed by [m_idx][n_idx][k_idx][trans_a][trans_b]
  //! Contains measured performance data for ARM Cortex-A76 processor
  extern const double gflops_table[M_SIZE][N_SIZE][K_SIZE][2][2];

}  // namespace einsum_ir::model::a76

#endif  // EINSUM_IR_MODEL_A76_BENCH_A76_H