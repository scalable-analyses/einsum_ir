//TODO change inlude guard
#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION_SCALAR_NEW
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION_SCALAR_NEW

#include "ContractionBackend.h"

namespace einsum_ir {
  namespace backend {
    class ContractionBackendScalar;
  }
}

class einsum_ir::backend::ContractionBackendScalar: public ContractionBackend {
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

    /**
     * Compiler-based multiply add kernel.
     *
     * @param_t T_LEFT data type of the left input.
     * @param_t T_RIGHT data type of the right input.
     * @param_t T_OUT data type of the output.
     **/
    template < typename T_LEFT,
               typename T_RIGHT,
               typename T_OUT >
    static void kernel_madd( void const * i_in_left,
                             void const * i_in_right,
                             void       * io_out );

    //! first-touch kernel
    void (* m_kernel_first_touch)( void const *,
                                   void       * ) = nullptr;

    //! main kernel
    void (* m_kernel_main)( void const *,
                            void const *,
                            void       * ) = nullptr;

    //! last-touch kernel
    void (* m_kernel_last_touch)( void const *,
                                  void       * ) = nullptr;

  public:
    /**
     * Executes the first touch kernel on the given data section of the tensor.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_first_touch( void const * i_out_aux,
                             void       * io_out );

    /**
     * Executes the main kernel on the given data sections of the tensors.
     *
     * @param i_left pointer to a data section of the left tensor.
     * @param i_right pointer to a data section of the right tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_main( void const * i_left,
                      void const * i_right,
                      void       * io_out );

    /**
     * Executes the last touch kernel on the given data section of the tensor.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_last_touch( void const * i_out_aux,
                            void       * io_out );

    /**
     * Compiles all kernels
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile_kernels();
};

#endif
