#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION_TPP
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION_TPP

#include <libxsmm.h>
#include "BinaryContraction.h"
#include "ContractionLoopsTpp.h"

namespace einsum_ir {
  namespace backend {
    class BinaryContractionTpp;
  }
}

class einsum_ir::backend::BinaryContractionTpp: public BinaryContraction {
  private:
    //! LIBXSMM-based unary first-touch TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_first_touch_unary = nullptr;

    //! LIBXSMM-based binary first-touch TPP
    libxsmm_meltwfunction_binary m_xmm_kernel_first_touch_binary = nullptr;

    //! LIBXSMM-based main TPP which is called in the innermost loop
    libxsmm_xmmfunction m_xmm_kernel_main;

    //! LIBXSMM-based unary last-touch TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_last_touch_unary = nullptr;

    //! LIBXSMM-based binary last-touch TPP
    libxsmm_meltwfunction_binary m_xmm_kernel_last_touch_binary = nullptr;

    //! packing
    ContractionPackingTpp * m_packing = nullptr;

    //! contraction loop interface
    ContractionLoopsTpp m_cont_loops;

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

    /**
     * Converts the given native datatype to a LIBXSMM datatype.
     *
     * @param i_data_type native datatype.
     * @return corresponding LIBXSMM datatype.
     */
    static libxsmm_datatype dtype_to_libxsmm( data_t i_dtype );

  public:
    /**
     * Destructor
     **/
    ~BinaryContractionTpp();

    /**
     * Compiles the binary contraction.
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
