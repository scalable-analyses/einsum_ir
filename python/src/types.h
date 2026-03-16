#ifndef EINSUM_IR_PY_TYPES_H
#define EINSUM_IR_PY_TYPES_H

#include <cstdint>

namespace einsum_ir {
  namespace py {
    /// primitive type
    enum class prim_t : uint32_t {
      none      =  0,
      zero      =  1,
      copy      =  2,
      relu      =  3,
      gemm      =  4,
      brgemm    =  5,
      undefined = 99
    };

    /// dimension type
    enum class dim_t : uint32_t {
      c         = 0, 
      m         = 1, 
      n         = 2, 
      k         = 3, 
      undefined = 99
    };

    /// execution type
    enum class exec_t : uint32_t {
      seq       = 0, 
      prim      = 1,
      shared    = 2,
      sfc       = 3,
      undefined = 99
    };

    /// data type
    enum class dtype_t : uint32_t {
      fp32 = 0,
      fp64 = 1
    };
  }
}

#endif
