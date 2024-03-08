#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION_TBLIS
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION_TBLIS

#include "BinaryContraction.h"
#include <tblis/tblis.h>

namespace einsum_ir {
  namespace backend {
    class BinaryContractionTblis;
  }
}

class einsum_ir::backend::BinaryContractionTblis: public BinaryContraction {
  private:
    //! used tensor ordering
    tenord_t m_tensor_ordering = UNDEFINED_TENORD;

    //! tblis tensor descriptors
    tblis::tblis_tensor m_tblis_tensor_left;
    tblis::tblis_tensor m_tblis_tensor_right;
    tblis::tblis_tensor m_tblis_tensor_out;

    //! tblis strides
    std::vector< tblis::stride_type > m_tblis_strides_left;
    std::vector< tblis::stride_type > m_tblis_strides_right;
    std::vector< tblis::stride_type > m_tblis_strides_out;

    //! tblis sizes
    std::vector< tblis::len_type > m_tblis_sizes_left;
    std::vector< tblis::len_type > m_tblis_sizes_right;
    std::vector< tblis::len_type > m_tblis_sizes_out;

    //! tblis dimension ids
    std::vector< char > m_tblis_dim_ids_left;
    std::vector< char > m_tblis_dim_ids_right;
    std::vector< char > m_tblis_dim_ids_out;

  public:
    /**
     * Compiles the binary contraction.
     * @return SUCCESS if successful, error code otherwise.
     **/
    err_t compile();

    /**
     * Initializes the threading configuration of the contraction.
     *
     * @param i_num_tasks_target number of targeted tasks.
     **/
    void threading( int64_t i_num_tasks_target  );

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_left left input tensor.
     * @param i_tensor_right right input tensor.
     * @param io_tensor_out output tensor.
     **/
    void contract( void const * i_tensor_left,
                   void const * i_tensor_right,
                   void       * io_tensor_out );

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_left left input tensor.
     * @param i_tensor_right right input tensor.
     * @param i_tensor_out_aux auxiliary data w.r.t. output tensor.
     * @param io_tensor_out output tensor.
     **/
    void contract( void const * i_tensor_left,
                   void const * i_tensor_right,
                   void const * i_tensor_out_aux,
                   void       * io_tensor_out );
};

#endif
