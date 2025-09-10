#ifndef EINSUM_IR_BASIC_CONSTANTS
#define EINSUM_IR_BASIC_CONSTANTS

#include <cstdint>
#include <vector>

namespace einsum_ir {
  namespace basic {
    typedef enum {
      ZERO            = 0,
      RELU            = 1,
      ADD             = 2,
      COPY            = 3,
      MADD            = 4,
      CPX_ZERO        = 6,
      CPX_ADD         = 7,
      CPX_MADD        = 8,
      CPX_COPY        = 9,
      BR_MADD         = 12,
      PACKED_MADD     = 13,
      CPX_PACKED_MADD = 14,
      UNDEFINED_KTYPE = 99
    } kernel_t;

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
      SUCCESS                   =  0,
      COMPILATION_FAILED        =  1,
      INVALID_CPX_DIM           =  2,
      UNDEFINED_ERROR           = 99
    } err_t;

    typedef enum {
      FP32            = 0,
      FP64            = 1,
      UNDEFINED_DTYPE = 99
    } data_t;

    typedef enum {
      OMP =  0, // OMP parallelized dim
      SEQ =  1, // Sequential dimension
      SFC =  2, // SFC dimension (always is omp parallelized)
      PRIM = 3, // Primitive dimension 
      UNDEFINED_EXECTYPE = 99
    } exec_t;

    typedef enum {
      NONE           = 0, // no packed gemm
      ALL_STRIDE_ONE = 1, // all dimensions have stride one
      OUT_STRIDE_ONE = 2  // output dimension has stride one
    } packed_gemm_t;

    typedef uint8_t sfc_t;

    struct thread_info {
      int64_t   offset_left    = 0;
      int64_t   offset_right   = 0;
      int64_t   offset_out_aux = 0;
      int64_t   offset_out     = 0;
      char    * memory_left    = nullptr;
      char    * memory_right   = nullptr;

      std::vector<sfc_t>   movement_ids;
      std::vector<const char *> cached_ptrs_left;
      std::vector<const char *> cached_ptrs_right;
    };

    struct iter_property {
      dim_t   dim_type             = dim_t::UNDEFINED_DIM;
      exec_t  exec_type            = exec_t::SEQ;
      int64_t size                 = 0;
      int64_t stride_left          = 0;
      int64_t stride_right         = 0;
      int64_t stride_out_aux       = 0;
      int64_t stride_out           = 0;
      int64_t packing_stride_left  = 0;
      int64_t packing_stride_right = 0;
    };

    constexpr int64_t ce_n_bytes( data_t i_dtype ) {
      if(      i_dtype == FP32 )  return 4;
      else if( i_dtype == FP64 )  return 8;
      else                        return -1;
    }
  }
}

#endif