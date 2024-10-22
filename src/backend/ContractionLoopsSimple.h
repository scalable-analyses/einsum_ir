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
     * @param i_sizes sizes of the dimensions
     * @param i_strides_left strides of the left input tensor.
     * @param i_strides_right strides of the right input tensor.
     * @param i_strides_out_aux strides of the auxiliary output tensor.
     * @param i_strides_out strides of the output tensor.
     * @param i_dim_type the tpye of th dimension
     * @param i_num_bytes_scalar_left number of bytes per scalar in the left tensor.
     * @param i_num_bytes_scalar_right number of bytes per scalar in the right tensor.
     * @param i_num_bytes_scalar_out number of bytes per scalar in the output tensor.
     * @param i_kernel_first_touch first touch kernel, may be ignored by passing nullptr.
     * @param i_kernel_main main kernel which is applied in the innermost loop.
     * @param i_kernel_last_touch last touch kernel, may be ignored by passing nullptr.
     **/
    void init( std::map< int64_t, int64_t > const * i_sizes,
               std::map< int64_t, int64_t > const * i_strides_left,
               std::map< int64_t, int64_t > const * i_strides_right,
               std::map< int64_t, int64_t > const * i_strides_out_aux,
               std::map< int64_t, int64_t > const * i_strides_out,
               std::map< int64_t, dim_t >   const * i_dim_type,
               std::vector<int64_t>               * i_loop_ids,
               int64_t                              i_num_bytes_scalar_left,
               int64_t                              i_num_bytes_scalar_right,
               int64_t                              i_num_bytes_scalar_out,
               void                              (* i_kernel_first_touch)( void const *,
                                                                           void       * ),
               void                              (* i_kernel_main)( void const *,
                                                                    void const *,
                                                                    void       * ),
               void                              (* i_kernel_last_touch)( void const *,
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
