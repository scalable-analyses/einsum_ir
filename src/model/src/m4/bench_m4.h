#ifndef EINSUM_IR_MODEL_M4_BENCH_M4_H
#define EINSUM_IR_MODEL_M4_BENCH_M4_H

#include <cstddef>

namespace einsum_ir::model::m4 {

  //! Dimension sizes
  static const int M_SIZE = 48;
  static const int N_SIZE = 48;
  static const int K_SIZE = 6;

  //! Tested dimension values
  static const int M_VALUES[M_SIZE] = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65,
                                       79, 80, 81, 95, 96, 97, 111, 112, 113, 127, 128, 129,
                                       143, 144, 145, 159, 160, 161, 175, 176, 177, 191, 192, 193,
                                       207, 208, 209, 223, 224, 225, 239, 240, 241, 255, 256};

  static const int N_VALUES[N_SIZE] = {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65,
                                       79, 80, 81, 95, 96, 97, 111, 112, 113, 127, 128, 129,
                                       143, 144, 145, 159, 160, 161, 175, 176, 177, 191, 192, 193,
                                       207, 208, 209, 223, 224, 225, 239, 240, 241, 255, 256};

  static const int K_VALUES[K_SIZE] = {4, 16, 48, 128, 256, 512};

  //! GFLOPS table indexed by [m_idx][n_idx][k_idx][trans_b]
  extern const double gflops_table[M_SIZE][N_SIZE][K_SIZE][2];

}  // namespace einsum_ir::model::m4
#endif  // EINSUM_IR_MODEL_M4_BENCH_M4_H