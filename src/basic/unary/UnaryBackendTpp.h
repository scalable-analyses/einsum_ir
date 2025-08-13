#ifndef EINSUM_IR_BASIC_UNARY_BACKEND_TPP
#define EINSUM_IR_BASIC_UNARY_BACKEND_TPP

#include <libxsmm.h>
#include "UnaryBackend.h"

namespace einsum_ir {
  namespace basic {
    class UnaryBackendTpp;
  }
}

class einsum_ir::basic::UnaryBackendTpp: public UnaryBackend {
  private:

    //! LIBXSMM-based unary TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_unary = nullptr;

    //! LIBXSMM-based binary TPP
    libxsmm_meltwfunction_binary m_xmm_kernel_binary = nullptr;


    /**
     * coverts internal datatypes to libxsmm datatypes
     *
     * @return libxsmm datatype.
     **/
    libxsmm_datatype dtype_to_libxsmm( data_t i_dtype );



    /**
     * Kernel called in the innermost loop.
     *
     * @param i_in pointer to a data section of the tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_main( void const * i_in,
                              void       * io_out );

    /**
     * Compiles all kernels
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile_kernels();
};

#endif