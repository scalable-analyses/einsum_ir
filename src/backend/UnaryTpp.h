#ifndef EINSUM_IR_BACKEND_UNARY_TPP
#define EINSUM_IR_BACKEND_UNARY_TPP

#include <vector>
#include "Unary.h"
#include "../etops/unary/UnaryBackendTpp.h"

namespace einsum_ir {
  namespace backend {
    class UnaryTpp;
  }
}

class einsum_ir::backend::UnaryTpp: public Unary {
  private:
    //! unary tpp backend
    etops::UnaryBackendTpp m_backend;

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
