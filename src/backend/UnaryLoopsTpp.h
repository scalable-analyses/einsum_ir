#ifndef EINSUM_IR_BACKEND_UNARY_LOOPS_TPP
#define EINSUM_IR_BACKEND_UNARY_LOOPS_TPP

#include <libxsmm.h>
#include "UnaryLoops.h"

namespace einsum_ir {
  namespace backend {
    class UnaryLoopsTpp;
  }
}

class einsum_ir::backend::UnaryLoopsTpp: public UnaryLoops {
  private:
    //! LIBXSMM-based unary main TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_main = nullptr;

  public:
    /**
     * Initializes the TPP- and loop-based implementation of unary operators.
     *
     * @param i_num_dims number of dimensions.
     * @param i_sizes sizes of the dimensions.
     * @param i_strides_in strides of the input tensor.
     * @param i_strides_out strides of the output tensor.
     * @param i_num_bytes_in number of bytes per scalar in the input tensor.
     * @param i_num_bytes_out number of bytes per scalar in the output tensor.
     * @param i_kernel_main main kernel which is applied in the innermost loop.
     */
    void init( int64_t                             i_num_dims,
               int64_t                     const * i_sizes,
               int64_t                     const * i_strides_in,
               int64_t                     const * i_strides_out,
               int64_t                             i_num_bytes_in,
               int64_t                             i_num_bytes_out,
               libxsmm_meltwfunction_unary const   i_kernel_main );

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
