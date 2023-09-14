#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS_SIMPLE
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS_SIMPLE

#include "ContractionLoops.h"

namespace einsum_ir {
  namespace backend {
    class ContractionLoopsSimple;
  }
}

class einsum_ir::backend::ContractionLoopsSimple: public ContractionLoops {
  private:
    //! pointer-based first touch kernel
    void (* m_kernel_first_touch)( void const *, void * );

    //! pointer-based inner kernel
    void (*m_kernel_main)( void const *,
                           void const *,
                           void       * );

    //! pointer-based last touch kernel
    void (* m_kernel_last_touch)( void const *, void * );

  public:
    /**
     * Initializes the simple contraction loops.
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
     * @param i_kernel_first_touch first touch kernel, may be ignored by passing nullptr.
     * @param i_kernel_main main kernel which is applied in the innermost loop.
     * @param i_kernel_last_touch last touch kernel, may be ignored by passing nullptr.
     **/
    void init( int64_t         i_num_dims_c,
               int64_t         i_num_dims_m,
               int64_t         i_num_dims_n,
               int64_t         i_num_dims_k,
               int64_t const * i_sizes_c,
               int64_t const * i_sizes_m,
               int64_t const * i_sizes_n,
               int64_t const * i_sizes_k,
               int64_t const * i_strides_in_left_c,
               int64_t const * i_strides_in_left_m,
               int64_t const * i_strides_in_left_k,
               int64_t const * i_strides_in_right_c,
               int64_t const * i_strides_in_right_n,
               int64_t const * i_strides_in_right_k,
               int64_t const * i_strides_out_aux_c,
               int64_t const * i_strides_out_aux_m,
               int64_t const * i_strides_out_aux_n,
               int64_t const * i_strides_out_c,
               int64_t const * i_strides_out_m,
               int64_t const * i_strides_out_n,
               int64_t         i_num_bytes_scalar_left,
               int64_t         i_num_bytes_scalar_right,
               int64_t         i_num_bytes_scalar_out,
               void         (* i_kernel_first_touch)( void const *,
                                                      void       * ),
               void         (* i_kernel_main)( void const *,
                                               void const *,
                                               void       * ),
               void         (* i_kernel_last_touch)( void const *,
                                                     void       * ) );

    /**
     * Executes the first touch kernel on the given data section of the output tensor.
     *
     * @param i_out_aux pointer to the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_first_touch( void const * i_out_aux,
                             void       * io_out );

    /**
     * Executes the main kernel on the given data section of the tensors.
     *
     * @param i_left pointer to a data section of the left tensor.
     * @param i_right pointer to a data section of the right tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_main( void const * i_left,
                      void const * i_right,
                      void       * io_out );

    /**
     * Executes the last touch kernel on the given data section of the output tensor.
     *
     * @param i_out_aux pointer to the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_last_touch( void const * i_out_aux,
                            void       * io_out );
};

#endif
