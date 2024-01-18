#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION_BLAS
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION_BLAS

#include "BinaryContraction.h"
#include "ContractionLoopsBlas.h"

namespace einsum_ir {
  namespace backend {
    class BinaryContractionBlas;
  }
}

class einsum_ir::backend::BinaryContractionBlas: public BinaryContraction {
  private:
    //! contraction loop interface
    ContractionLoopsBlas m_cont_loops;

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
