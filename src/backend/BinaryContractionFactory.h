#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION_FACTORY
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION_FACTORY

#include <vector>
#include "BinaryContraction.h"

namespace einsum_ir {
  namespace backend {
    class BinaryContractionFactory;
  }
}

class einsum_ir::backend::BinaryContractionFactory {
  public:
    /**
     * Checks if the given backend is supported.
     *
     * @param i_backend backend type which should be checked.
     * @return true if the backend is supported, false otherwise.
     **/
    static bool supports( backend_t i_backend );

    /**
     * Create a binary contraction using the given backend.
     * If the backend is not supported, a nullptr is returned.
     *
     * @param i_backend used backend.
     * @return new binary contraction.
     **/
    static BinaryContraction * create( backend_t i_backend );
};


#endif