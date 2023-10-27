#ifndef EINSUM_IR_BACKEND_UNARY_LOOPS_SIMPLE
#define EINSUM_IR_BACKEND_UNARY_LOOPS_SIMPLE

#include "UnaryLoops.h"

namespace einsum_ir {
  namespace backend {
    class UnaryLoopsSimple;
  }
}

class einsum_ir::backend::UnaryLoopsSimple: public UnaryLoops {
  private:
    //! pointer-based main kernel
    void (*m_kernel_main)( void const *,
                           void       * );

  public:
    /**
     * Initializes the loop-based implementation of unary operators.
     *
     * @param i_num_dims number of dimensions.
     * @param i_sizes sizes of the dimensions.
     * @param i_strides_in strides of the input tensor.
     * @param i_strides_out strides of the output tensor.
     * @param i_num_bytes_in number of bytes per scalar in the input tensor.
     * @param i_num_bytes_out number of bytes per scalar in the output tensor.
     * @param i_kernel_main main kernel which is applied in the innermost loop.
     */
    void init( int64_t         i_num_dims,
               int64_t const * i_sizes,
               int64_t const * i_strides_in,
               int64_t const * i_strides_out,
               int64_t         i_num_bytes_in,
               int64_t         i_num_bytes_out,
               void         (* i_kernel_main)( void const *,
                                               void       * ) );

    /**
     * Executes the main kernel on the given data section of the tensors.
     *
     * @param i_ptr_in pointer to a data section of the input tensor.
     * @param io_ptr_out pointer to a data section of the output tensor.
     **/
    void kernel_main( void const * i_ptr_in,
                      void       * io_ptr_out );
};

#endif
