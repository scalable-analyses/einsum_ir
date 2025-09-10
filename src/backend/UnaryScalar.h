#ifndef EINSUM_IR_BACKEND_UNARY_SCALAR
#define EINSUM_IR_BACKEND_UNARY_SCALAR

#include "Unary.h"
#include "../basic/unary/UnaryBackendScalar.h"

namespace einsum_ir {
  namespace backend {
    class UnaryScalar;
  }
}

class einsum_ir::backend::UnaryScalar: public Unary {
  private:
    //! unary scalar backend
    basic::UnaryBackendScalar m_backend;

  public:
    /**
     * Compiles the unary operation.
     *
     * @return SUCCESS if successful, error code otherwise.
     **/
    err_t compile();

    /**
     * Evaluates the unary operation on the given data.
     *
     * @param i_tensor_in input tensor.
     * @param io_tensor_out output tensor.
     **/
    void eval( void const * i_tensor_in,
               void       * io_tensor_out );
};

#endif
