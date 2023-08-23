#ifndef EINSUM_IR_CONSTANTS
#define EINSUM_IR_CONSTANTS

#include <cstdint>

namespace einsum_ir {
  typedef enum {
    C = 0, // left, right, out
    M = 1, // left, out
    N = 2, // right, out
    K = 3, // left, right
    I = 4, // left
    J = 5, // right
    UNDEFINED_DIM = 99
  } dim_t;

  typedef enum {
    LEFT_NATIVE_RIGHT_NATIVE_OUT_NATIVE,
    LEFT_BC_BM_BK_RIGHT_BC_BN_BK_OUT_NATIVE,
    LEFT_BC_BM_BK_KB_MB_RIGHT_BC_BN_BK_NB_KB_OUT_NATIVE,
    LEFT_BC_BM_BK_KB_MB_CB_RIGHT_BC_BN_BK_NB_KB_CB_OUT_NATIVE,
    UNDEFINED_TENORD = 99
  } tenord_t;

  typedef enum {
    SUCCESS = 0,
    COMPILATION_FAILED = 1,
    DIMENSION_ORDERING_FAILED = 2
  } err_t;

  typedef enum {
    FP32 = 0,
    FP64 = 1,
    UNDEFINED_DTYPE = 99
  } data_t;

  typedef enum {
    ZERO,
    RELU,
    MADD,
    CUSTOM_KTYPE,
    UNDEFINED_KTYPE = 99
  } kernel_t;

  constexpr int64_t ce_n_bytes( data_t i_dtype ) {
    if( i_dtype == FP32 ) return 4;
    else if( i_dtype == FP64 ) return 8;
    return -1;
  }
}

#endif