#ifndef EINSUM_IR_CONSTANTS
#define EINSUM_IR_CONSTANTS

#include <cstdint>
#include <basic/constants.h>

namespace einsum_ir {
  typedef enum {
    C = 0,  // left, right, out
    M = 1,  // left, out
    N = 2,  // right, out
    K = 3,  // left, right
    CPX = 4,// Complex dimension in left, right, out
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

  constexpr basic::dim_t ce_dimt_to_basic( dim_t i_dim ) {
    if(      i_dim == dim_t::C   ) return basic::dim_t::C;
    else if( i_dim == dim_t::M   ) return basic::dim_t::M;
    else if( i_dim == dim_t::N   ) return basic::dim_t::N;
    else if( i_dim == dim_t::K   ) return basic::dim_t::K;
    else if( i_dim == dim_t::CPX ) return basic::dim_t::CPX;
    else if( i_dim == dim_t::I   ) return basic::dim_t::I;
    else if( i_dim == dim_t::J   ) return basic::dim_t::J;
    else return basic::dim_t::UNDEFINED_DIM;
  }

  constexpr basic::data_t ce_dtype_to_basic( data_t i_dtype ) {
    if(      i_dtype == FP32 ) return basic::data_t::FP32;
    else if( i_dtype == FP64 ) return basic::data_t::FP64;
    else                       return basic::data_t::UNDEFINED_DTYPE;
  }

  constexpr basic::kernel_t ce_kernelt_to_basic( kernel_t i_ktype ) {
    if(      i_ktype == ZERO            ) return basic::kernel_t::ZERO;
    else if( i_ktype == RELU            ) return basic::kernel_t::RELU;
    else if( i_ktype == ADD             ) return basic::kernel_t::ADD;
    else if( i_ktype == COPY            ) return basic::kernel_t::COPY;
    else if( i_ktype == MADD            ) return basic::kernel_t::MADD;
    else if( i_ktype == CPX_ZERO        ) return basic::kernel_t::CPX_ZERO;
    else if( i_ktype == CPX_ADD         ) return basic::kernel_t::CPX_ADD;
    else if( i_ktype == CPX_MADD        ) return basic::kernel_t::CPX_MADD;
    else if( i_ktype == CPX_COPY        ) return basic::kernel_t::CPX_COPY;
    else if( i_ktype == BR_MADD         ) return basic::kernel_t::BR_MADD;
    else if( i_ktype == PACKED_MADD     ) return basic::kernel_t::PACKED_MADD;
    else if( i_ktype == CPX_PACKED_MADD ) return basic::kernel_t::CPX_PACKED_MADD;
    else                                  return basic::kernel_t::UNDEFINED_KTYPE;
  }

  constexpr err_t ce_basic_err_to_err( basic::err_t i_err ) {
    if(      i_err == basic::err_t::SUCCESS                   ) return err_t::SUCCESS;
    else if( i_err == basic::err_t::COMPILATION_FAILED        ) return err_t::COMPILATION_FAILED;
    else if( i_err == basic::err_t::INVALID_CPX_DIM           ) return err_t::INVALID_CPX_DIM;
    else                                                        return err_t::UNDEFINED_ERROR;
  }

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