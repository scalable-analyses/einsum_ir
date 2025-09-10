#ifndef EINSUM_IR_BASIC_UNARY_BACKEND_SCALAR
#define EINSUM_IR_BASIC_UNARY_BACKEND_SCALAR

#include "UnaryBackend.h"

namespace einsum_ir {
  namespace basic {
    class UnaryBackendScalar;
  }
}

class einsum_ir::basic::UnaryBackendScalar: public UnaryBackend {
  private:
    /**
     * Compiler-based zero kernel.
     *
     * @param_t datatype.
     * @param o_data data which is zeroed.
     **/
    template < typename T >
    static void kernel_zero( void const *,
                             void       * o_data );

    /**
     * Compiler-based ReLU kernel.
     *
     * @param_t datatype.
     * @param io_data data to which the ReLU is applied.
     **/
    template < typename T >
    static void kernel_relu( void const *,
                             void       * io_data );

    /**
     * Compiler-based copy kernel.
     *
     * @param_t datatype.
     * @param i_data_src source of the copy operation.
     * @param i_data_dst destination of the copy operation.
     **/
    template < typename T >
    static void kernel_copy( void const * i_data_src,
                             void       * io_data_dst );

    //! main kernel
    void (* m_kernel)( void const *,
                       void       * ) = nullptr;


  public:
    /**
     * Executes the main kernel on the given data sections of the tensors.
     *
     * @param i_in pointer to a data section of the input tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_main( void const * i_in,
                      void       * io_out );

    /**
     * Compiles all kernels
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile_kernels();
};

#endif
