#ifndef EINSUM_IR_BINARY_CONTRACTION_BACKEND_TPP
#define EINSUM_IR_BINARY_CONTRACTION_BACKEND_TPP

#include <libxsmm.h>
#include "ContractionBackend.h"

namespace einsum_ir {
  namespace binary {
    class ContractionBackendTpp;
  }
}

class einsum_ir::binary::ContractionBackendTpp: public ContractionBackend {
  private:

    //! LIBXSMM-based unary first-touch TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_first_touch_unary = nullptr;

    //! LIBXSMM-based binary first-touch TPP
    libxsmm_meltwfunction_binary m_xmm_kernel_first_touch_binary = nullptr;

    //! LIBXSMM-based main TPP
    libxsmm_gemmfunction m_xmm_kernel_main = nullptr;

    //! LIBXSMM-based unary last-touch TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_last_touch_unary = nullptr;

    //! LIBXSMM-based binary last-touch TPP
    libxsmm_meltwfunction_binary m_xmm_kernel_last_touch_binary = nullptr;

    /**
     * coverts internal datatypes to libxsmm datatypes
     *
     * @return libxsmm datatype.
     **/
    libxsmm_datatype dtype_to_libxsmm( data_t i_dtype );
    
  public:
    /**
     * Kernel applied to the output tensor before the main primitive touches the memory.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_first_touch( void const * i_out_aux,
                             void       * io_out );

    /**
     * Kernel applied to the output tensor after the main primitve finished using the memory.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_last_touch( void const * i_out_aux,
                            void       * io_out );

    /**
     * Kernel called in the innermost loop.
     *
     * @param i_left pointer to a data section of the left tensor.
     * @param i_right pointer to a data section of the right tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_main( void const * i_left,
                              void const * i_right,
                              void       * io_out );

    /**
     * Compiles all kernels
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile_kernels();
};

#endif