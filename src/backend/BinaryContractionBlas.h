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

    //! BC sizes
    std::vector< int64_t > m_sizes_bc;
    //! BM sizes
    std::vector< int64_t > m_sizes_bm;
    //! BN sizes
    std::vector< int64_t > m_sizes_bn;
    //! BK sizes
    std::vector< int64_t > m_sizes_bk;

    //! BC strides of of the left tensor
    std::vector< int64_t > m_strides_left_bc;
    //! BM strides of the left tensor
    std::vector< int64_t > m_strides_left_bm;
    //! BK strides of the left tensor
    std::vector< int64_t > m_strides_left_bk;
    //! BI strides of the left tensor
    std::vector< int64_t > m_strides_left_bi;

    //! BC strides of the right tensor
    std::vector< int64_t > m_strides_right_bc;
    //! BN strides of the right tensor
    std::vector< int64_t > m_strides_right_bn;
    //! BK strides of the right tensor
    std::vector< int64_t > m_strides_right_bk;
    //! BJ strides of the right tensor
    std::vector< int64_t > m_strides_right_bj;

    //! BC strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_bc;
    //! BM strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_bm;
    //! BN strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_bn;

    //! BC strides of the output tensor
    std::vector< int64_t > m_strides_out_bc;
    //! BM strides of the output tensor
    std::vector< int64_t > m_strides_out_bm;
    //! BN strides of the output tensor
    std::vector< int64_t > m_strides_out_bn;

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
