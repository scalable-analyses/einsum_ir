#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS_TPP
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS_TPP

#include "ContractionLoops.h"
#include "ContractionPackingTpp.h"
#include <libxsmm.h>

namespace einsum_ir {
  namespace backend {
    class ContractionLoopsTpp;
  }
}

class einsum_ir::backend::ContractionLoopsTpp: public ContractionLoops {
  private:
    //! LIBXSMM-based unary first-touch TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_first_touch_unary = nullptr;

    //! LIBXSMM-based binary first-touch TPP
    libxsmm_meltwfunction_binary m_xmm_kernel_first_touch_binary = nullptr;

    //! LIBXSMM-based main TPP
    libxsmm_gemmfunction m_xmm_kernel_main = nullptr;

    //! LIBXSMM-based unary last-touch TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_last_touch_unary = nullptr;

    //! LIBXSMM-based unary last-touch TPP
    libxsmm_meltwfunction_binary m_xmm_kernel_last_touch_binary = nullptr;

  public:
    /**
     * Initializes the TPP-based contraction loops.
     *
     * Shortcuts:
     *   C: batch dimensions which appears in all tensors.
     *   M: dimensions appear in left input and output.
     *   N: dimensions appear in right input and output.
     *   K: reduction dimensions which appear in both inputs,
     *
     * @param i_num_dims_c number of C dimensions.
     * @param i_num_dims_m number of M dimensions.
     * @param i_num_dims_n number of N dimensions.
     * @param i_num_dims_k number of K dimensions.
     * @param i_dim_ids_c dimensiom ids of the C dimensions.
     * @param i_dim_ids_m dimensiom ids of the M dimensions.
     * @param i_dim_ids_n dimensiom ids of the N dimensions.
     * @param i_dim_ids_k dimensiom ids of the K dimensions.
     * @param i_sizes sizes of the dimensions
     * @param i_strides_left strides of the left input tensor.
     * @param i_strides_right strides of the right input tensor.
     * @param i_strides_out_aux strides of the auxiliary output tensor.
     * @param i_strides_out strides of the output tensor.
     * @param i_dim_type the tpye of th dimension
     * @param i_num_bytes_scalar_left number of bytes per scalar in the left tensor.
     * @param i_num_bytes_scalar_right number of bytes per scalar in the right tensor.
     * @param i_num_bytes_scalar_out number of bytes per scalar in the output tensor.
     * @param i_ktype_first_touch first-touch kernel type.
     * @param i_ktype_main main kernel type.
     * @param i_ktype_last_touch last-touch kernel type.
     * @param i_xmm_kernel_first_touch_unary unary first-touch tpp.
     * @param i_xmm_kernel_first_touch_binary binary first-touch tpp.
     * @param i_xmm_kernel_main tpp which is applied in the innermost loop.
     * @param i_xmm_kernel_last_touch_unary unary last-touch tpp.
     * @param i_xmm_kernel_last_touch_binary binary last-touch tpp.
     * @param i_packing manages packing of inputs

     **/
    void init( int64_t                              i_num_dims_c,
               int64_t                              i_num_dims_m,
               int64_t                              i_num_dims_n,
               int64_t                              i_num_dims_k,
               int64_t                      const * i_dim_ids_c,
               int64_t                      const * i_dim_ids_m,
               int64_t                      const * i_dim_ids_n,
               int64_t                      const * i_dim_ids_k,
               std::map< int64_t, int64_t > const * i_sizes,
               std::map< int64_t, int64_t > const * i_strides_left,
               std::map< int64_t, int64_t > const * i_strides_right,
               std::map< int64_t, int64_t > const * i_strides_out_aux,
               std::map< int64_t, int64_t > const * i_strides_out,
               std::map< int64_t, dim_t >   const * i_dim_type,
               int64_t                              i_num_bytes_scalar_left,
               int64_t                              i_num_bytes_scalar_right,
               int64_t                              i_num_bytes_scalar_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch,
               libxsmm_meltwfunction_unary  const   i_xmm_kernel_first_touch_unary,
               libxsmm_meltwfunction_binary const   i_xmm_kernel_first_touch_binary,
               libxsmm_gemmfunction         const   i_xmm_kernel_main,
               libxsmm_meltwfunction_unary  const   i_xmm_kernel_last_touch_unary,
               libxsmm_meltwfunction_binary const   i_xmm_kernel_last_touch_binary,
               ContractionPackingTpp              * i_packing );

    /**
     * Executes the first touch kernel on the given data section of the tensor.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_first_touch( void const * i_out_aux,
                             void       * io_out );

    /**
     * Executes the main tpp on the given data sections of the tensors.
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
};

#endif
