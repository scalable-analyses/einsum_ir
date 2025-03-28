#ifndef BINARY_BACKEND_CONSTANTS_LOCAL
#define BINARY_BACKEND_CONSTANTS_LOCAL

namespace einsum_ir {
  namespace backend{
    struct loop_property {
      einsum_ir::dim_t  dim_type  = einsum_ir::UNDEFINED_DIM;
      einsum_ir::exec_t exec_type = einsum_ir::SEQ;
      int64_t size                = 0;
      int64_t stride_left         = 0;
      int64_t stride_right        = 0;
      int64_t stride_out_aux      = 0;
      int64_t stride_out          = 0;
    };
  }
}

#endif