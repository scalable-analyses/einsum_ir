#ifndef BINARY_BACKEND_CONSTANTS_LOCAL
#define BINARY_BACKEND_CONSTANTS_LOCAL

#include <cstdint>
#include "../../constants.h"

namespace einsum_ir {
  namespace backend{
    typedef enum {
      OMP =  0, // OMP parallelized dim
      SEQ =  1, // Sequential dimension
      SFC =  2, // SFC dimnesion (always is also omp parallelized)
      PRIM = 3, // Primitive dimension 
      UNDEFINED_EXECTYPE = 99
    } exec_t;

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