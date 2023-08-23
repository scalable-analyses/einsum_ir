#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION_TPP
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION_TPP

#include "BinaryContraction.h"
#include <libxsmm.h>

namespace einsum_ir {
  namespace backend {
    class BinaryContractionTpp;
  }
}

class einsum_ir::backend::BinaryContractionTpp: public BinaryContraction {
  private:
    //! LIBXSMM-based first-touch TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_first_touch = nullptr;

    //! LIBXSMM-based TPP which is called in the innermost loop
    libxsmm_xmmfunction m_xmm_kernel_inner;

    //! LIBXSMM-based last-touch TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_last_touch = nullptr;

    //! used tensor ordering
    tenord_t m_tensor_ordering = UNDEFINED_TENORD;

    //! number of blocked C dimensions
    int64_t m_num_dims_cb = 0;
    //! number of blocked M dimensions
    int64_t m_num_dims_mb = 0;
    //! number of blocked N dimensions
    int64_t m_num_dims_nb = 0;
    //! number of blocked K dimensions
    int64_t m_num_dims_kb = 0;

    /**
     * Converts the given native datatype to a LIBXSMM datatype.
     *
     * @param i_data_type native datatype.
     * @return corresponding LIBXSMM datatype.
     */
    static libxsmm_datatype dtype_to_libxsmm( data_t i_dtype );

  public:
    /**
     * Compiles the binary contraction.
     **/
    err_t compile();

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_in_left left input tensor.
     * @param i_tensor_in_right right input tensor.
     * @param io_tensor_out output tensor. 
     **/
    void contract( void const * i_tensor_in_left,
                   void const * i_tensor_in_right,
                   void       * io_tensor_out );
};

#endif
