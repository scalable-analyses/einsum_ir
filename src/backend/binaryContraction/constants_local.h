#ifndef BINARY_BACKEND_CONSTANTS_LOCAL
#define BINARY_BACKEND_CONSTANTS_LOCAL

namespace einsum_ir {
  namespace backend{
    typedef enum {
      OMP =  0, // OMP parallelized dim
      SEQ =  1, // Sequential dimension
      SFC =  2, // SFC dimnesion (always is also omp parallelized)
      PRIM = 3, // Primitive dimension 
      UNDEFINED_EXECTYPE = 99
    } exec_t;

    typedef enum {
      MADD                 =  0,
      BR_MADD              =  1,
      PACKED_MADD          =  2,
      UNDEFINED_MAIN_KTYPE = 99
    } kernel_main_t;

    typedef enum {
      ZERO                =  0,
      RELU                =  1,
      ADD                 =  2,
      COPY                =  3,
      UNDEFINED_SUB_KTYPE = 99
    } kernel_sub_t;
    
    struct loop_property {
      dim_t  dim_type  = dim_t::UNDEFINED_DIM;
      exec_t exec_type = exec_t::SEQ;
      int64_t size                = 0;
      int64_t stride_left         = 0;
      int64_t stride_right        = 0;
      int64_t stride_out_aux      = 0;
      int64_t stride_out          = 0;
    };
  }
}

#endif