#ifndef EINSUM_IR_CONSTANTS
#define EINSUM_IR_CONSTANTS

#include <cstdint>
#include <etops/constants.h>

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

  constexpr etops::dim_t ce_dimt_to_etops( dim_t i_dim ) {
    if(      i_dim == dim_t::C   ) return etops::dim_t::C;
    else if( i_dim == dim_t::M   ) return etops::dim_t::M;
    else if( i_dim == dim_t::N   ) return etops::dim_t::N;
    else if( i_dim == dim_t::K   ) return etops::dim_t::K;
    else if( i_dim == dim_t::CPX ) return etops::dim_t::CPX;
    else if( i_dim == dim_t::I   ) return etops::dim_t::I;
    else if( i_dim == dim_t::J   ) return etops::dim_t::J;
    else return etops::dim_t::UNDEFINED_DIM;
    
  }

  constexpr etops::data_t ce_dtype_to_etops( data_t i_dtype ) {
    if(      i_dtype == FP32 ) return etops::data_t::FP32;
    else if( i_dtype == FP64 ) return etops::data_t::FP64;
    else                       return etops::data_t::UNDEFINED_DTYPE;
  }

  constexpr etops::kernel_t ce_kernelt_to_etops( kernel_t i_ktype ) {
    if(      i_ktype == ZERO            ) return etops::kernel_t::ZERO;
    else if( i_ktype == RELU            ) return etops::kernel_t::RELU;
    else if( i_ktype == ADD             ) return etops::kernel_t::ADD;
    else if( i_ktype == COPY            ) return etops::kernel_t::COPY;
    else if( i_ktype == MADD            ) return etops::kernel_t::MADD;
    else if( i_ktype == CPX_INT_LOW     ) return etops::kernel_t::CPX_INT_LOW;
    else if( i_ktype == CPX_ZERO        ) return etops::kernel_t::CPX_ZERO;
    else if( i_ktype == CPX_ADD         ) return etops::kernel_t::CPX_ADD;
    else if( i_ktype == CPX_MADD        ) return etops::kernel_t::CPX_MADD;
    else if( i_ktype == CPX_COPY        ) return etops::kernel_t::CPX_COPY;
    else if( i_ktype == CPX_INT_HIGH    ) return etops::kernel_t::CPX_INT_HIGH;
    else if( i_ktype == CUSTOM_KTYPE    ) return etops::kernel_t::CUSTOM_KTYPE;
    else if( i_ktype == BR_MADD         ) return etops::kernel_t::BR_MADD;
    else if( i_ktype == PACKED_MADD     ) return etops::kernel_t::PACKED_MADD;
    else if( i_ktype == CPX_PACKED_MADD ) return etops::kernel_t::CPX_PACKED_MADD;
    else                                 return etops::kernel_t::UNDEFINED_KTYPE;
  }

  constexpr err_t ce_etops_err_to_err( etops::err_t i_err ) {
    if(      i_err == etops::err_t::SUCCESS                   ) return err_t::SUCCESS;
    else if( i_err == etops::err_t::COMPILATION_FAILED        ) return err_t::COMPILATION_FAILED;
    else if( i_err == etops::err_t::DIMENSION_ORDERING_FAILED ) return err_t::DIMENSION_ORDERING_FAILED;
    else if( i_err == etops::err_t::TENSOR_BLOCKING_FAILED    ) return err_t::TENSOR_BLOCKING_FAILED;
    else if( i_err == etops::err_t::NO_DATA_PTR_PROVIDED      ) return err_t::NO_DATA_PTR_PROVIDED;
    else if( i_err == etops::err_t::CALLED_BEFORE_COMPILATION ) return err_t::CALLED_BEFORE_COMPILATION;
    else if( i_err == etops::err_t::INVALID_ID                ) return err_t::INVALID_ID;
    else if( i_err == etops::err_t::INVALID_BACKEND           ) return err_t::INVALID_BACKEND;
    else if( i_err == etops::err_t::INVALID_CPX_DIM           ) return err_t::INVALID_CPX_DIM;
    else if( i_err == etops::err_t::INVALID_DTYPE             ) return err_t::INVALID_DTYPE;
    else if( i_err == etops::err_t::INVALID_KTYPE             ) return err_t::INVALID_KTYPE;
    else                                                       return err_t::UNDEFINED_ERROR;
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