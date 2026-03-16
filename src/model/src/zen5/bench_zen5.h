#ifndef EINSUM_IR_MODEL_ZEN5_BENCH_ZEN5_H
#define EINSUM_IR_MODEL_ZEN5_BENCH_ZEN5_H

#include <cstddef>

namespace einsum_ir::model::zen5 {

  //! Dimension sizes
  static const int M_SIZE = 24;
  static const int N_SIZE = 11;
  static const int K_SIZE = 6;

  //! Tested dimension values
  static const int M_VALUES[M_SIZE] = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65,
                                       79, 80, 81, 95, 96, 97, 111, 112, 113, 127, 128};

  static const int N_VALUES[N_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 64, 256};

  static const int K_VALUES[K_SIZE] = {4, 16, 32, 48, 64, 128};

  //! GFLOPS table indexed by [m_idx][n_idx][k_idx][trans_a][trans_b]
  extern const double gflops_table[M_SIZE][N_SIZE][K_SIZE][2][2];

}  // namespace einsum_ir::model::zen5

#endif  // EINSUM_IR_MODEL_ZEN5_BENCH_ZEN5_H