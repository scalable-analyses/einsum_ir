#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS_TPP
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS_TPP

#include "ContractionLoops.h"
#include <libxsmm.h>
#include "UnaryTpp.h"

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

    // LIBXSMM-based unary for packing of left input
    UnaryTpp * m_unary_packing_left = nullptr;

    // LIBXSMM-based unary for packing of right input
    UnaryTpp * m_unary_packing_right = nullptr;

  public:
    /**
     * Destructor.
     **/
    ~ContractionLoopsTpp();

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
     * @param i_sizes_c sizes of the C dimensions.
     * @param i_sizes_m sizes of the M dimensions.
     * @param i_sizes_n sizes of the N dimensions.
     * @param i_sizes_k sizes of the K dimensions.
     * @param i_strides_in_left_c C strides of the left input tensor.
     * @param i_strides_in_left_m M strides of the left input tensor.
     * @param i_strides_in_left_k K strides of the left input tensor.
     * @param i_strides_in_right_c C strides of the right input tensor.
     * @param i_strides_in_right_n N strides of the right input tensor.
     * @param i_strides_in_right_k K strides of the right input tensor.
     * @param i_strides_out_aux_c C strides of the auxiliary output tensor.
     * @param i_strides_out_aux_m M strides of the auxiliary output tensor.
     * @param i_strides_out_aux_n N strides of the auxiliary output tensor.
     * @param i_strides_out_c C strides of the output tensor.
     * @param i_strides_out_m M strides of the output tensor.
     * @param i_strides_out_n N strides of the output tensor.
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
     * @param i_unary_packing_left unary packing kernel of left tensor
     * @param i_unary_packing_right unary packing kernel of right tensor
     * @param i_memory_packing_left memory for packing kernel of left tensor
     * @param i_memory_packing_right memory for packing kernel of rigth tensor
     **/
    void init( int64_t                             i_num_dims_c,
               int64_t                             i_num_dims_m,
               int64_t                             i_num_dims_n,
               int64_t                             i_num_dims_k,
               int64_t                     const * i_sizes_c,
               int64_t                     const * i_sizes_m,
               int64_t                     const * i_sizes_n,
               int64_t                     const * i_sizes_k,
               int64_t                     const * i_strides_in_left_c,
               int64_t                     const * i_strides_in_left_m,
               int64_t                     const * i_strides_in_left_k,
               int64_t                     const * i_strides_in_right_c,
               int64_t                     const * i_strides_in_right_n,
               int64_t                     const * i_strides_in_right_k,
               int64_t                     const * i_strides_out_aux_c,
               int64_t                     const * i_strides_out_aux_m,
               int64_t                     const * i_strides_out_aux_n,
               int64_t                     const * i_strides_out_c,
               int64_t                     const * i_strides_out_m,
               int64_t                     const * i_strides_out_n,
               int64_t                             i_num_bytes_scalar_left,
               int64_t                             i_num_bytes_scalar_right,
               int64_t                             i_num_bytes_scalar_out,
               kernel_t                            i_ktype_first_touch,
               kernel_t                            i_ktype_main,
               kernel_t                            i_ktype_last_touch,
               libxsmm_meltwfunction_unary  const  i_xmm_kernel_first_touch_unary,
               libxsmm_meltwfunction_binary const  i_xmm_kernel_first_touch_binary,
               libxsmm_gemmfunction         const  i_xmm_kernel_main,
               libxsmm_meltwfunction_unary  const  i_xmm_kernel_last_touch_unary,
               libxsmm_meltwfunction_binary const  i_xmm_kernel_last_touch_binary,
               UnaryTpp                          * i_unary_packing_left,
               UnaryTpp                          * i_unary_packing_right,
               int64_t                             i_memory_packing_left,
               int64_t                             i_memory_packing_right );

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
    
    /**
     * Kernel to pack the left input tensor of the main kernel.
     *
     * @param i_in  pointer to a data section of the input tensor.
     * @param i_out  pointer to output of packing.
     **/
    void kernel_pack_left( void * i_in,
                           void * io_out );
    
    /**
     * Kernel to pack the right input tensor of the main kernel.
     *
     * @param i_in  pointer to a data section of the input tensor.
     * @param i_out  pointer to output of packing.
     **/
    void kernel_pack_right( void * i_in,
                            void * io_out );
};

#endif
