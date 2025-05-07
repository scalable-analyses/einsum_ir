#ifndef EINSUM_IR_CONSTANTS
#define EINSUM_IR_CONSTANTS

#include <cstdint>

namespace einsum_ir {
  typedef enum {
    C = 0,  // left, right, out
    M = 1,  // left, out
    N = 2,  // right, out
    K = 3,  // left, right
    CPX = 4,// left, right, out
    I = 5,  // left
    J = 6,  // right
    UNDEFINED_DIM = 99
  } dim_t;

  typedef enum {
    LEFT_NATIVE_RIGHT_NATIVE_OUT_NATIVE                             = 0,
    LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE       = 1,
    LEFT_BC_BM_BK_BI_KB_MB_CB_RIGHT_BC_BN_BK_BJ_NB_KB_CB_OUT_NATIVE = 2,
    LEFT_BC_BM_BK_BI_CB_KB_MB_RIGHT_BC_BN_BK_BJ_CB_NB_KB_OUT_NATIVE = 3,
    UNDEFINED_TENORD                                                = 99
  } tenord_t;

  typedef enum {
    LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB = 0,
    LEFT_X_CB_KB_MB_RIGHT_X_CB_NB_KB_OUT_NB_X_MB_CB = 1,
    UNDEFINED_PRIMBLO                               = 99
  } primblo_t;

  typedef enum {
    SUCCESS                   =  0,
    COMPILATION_FAILED        =  1,
    DIMENSION_ORDERING_FAILED =  2,
    TENSOR_BLOCKING_FAILED    =  3,
    NO_DATA_PTR_PROVIDED      =  4,
    CALLED_BEFORE_COMPILATION =  5,
    INVALID_ID                =  6,
    INVALID_BACKEND           =  7,
    INVALID_CPX_DIM           =  8,
    INVALID_DTYPE             =  9,
    INVALID_KTYPE             = 10,
    UNDEFINED_ERROR           = 99
  } err_t;

  typedef enum {
    FP32            = 0,
    FP64            = 1,
    UNDEFINED_DTYPE = 99
  } data_t;

  typedef enum {
    REAL_ONLY       = 0,
    BATCH_INNER     = 1,
    BATCH_OUTER     = 2,
    UNDEFINED_CTYPE = 99
  } complex_t;

  typedef enum {
    ZERO            = 0,
    RELU            = 1,
    ADD             = 2,
    COPY            = 3,
    MADD            = 4,
    CPX_INT_LOW     = 5, // internal use only
    CPX_ZERO        = 6,
    CPX_ADD         = 7,
    CPX_MADD        = 8,
    CPX_COPY        = 9,
    CPX_INT_HIGH    = 10, // internal use only
    CUSTOM_KTYPE    = 11,
    BR_MADD         = 12,
    PACKED_MADD     = 13,
    CPX_PACKED_MADD = 14,
    UNDEFINED_KTYPE = 99
  } kernel_t;

  typedef enum {
    AUTO   = 0,
    SCALAR = 1,
    TPP    = 2,
    BLAS   = 3,
    TBLIS  = 4,
    UNDEFINED_BACKEND = 99
  } backend_t;

  constexpr int64_t ce_n_bytes( data_t i_dtype ) {
    if(      i_dtype == FP32 )  return 4;
    else if( i_dtype == FP64 )  return 8;
    else                        return -1;
  }

  constexpr bool ce_cpx_op( kernel_t i_ktype ) {
    if(    i_ktype > kernel_t::CPX_INT_LOW
        && i_ktype < kernel_t::CPX_INT_HIGH ) {
          return true;
    }
    else {
      return false;
    }
  }
}

#endif