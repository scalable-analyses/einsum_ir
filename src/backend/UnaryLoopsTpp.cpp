#include "UnaryLoopsTpp.h"

void einsum_ir::backend::UnaryLoopsTpp::init( int64_t                             i_num_dims,
                                              int64_t                     const * i_sizes,
                                              int64_t                     const * i_strides_in,
                                              int64_t                     const * i_strides_out,
                                              int64_t                             i_num_bytes_in,
                                              int64_t                             i_num_bytes_out,
                                              libxsmm_meltwfunction_unary const   i_kernel_main ) {
  einsum_ir::backend::UnaryLoops::init( i_num_dims,
                                        i_sizes,
                                        i_strides_in,
                                        i_strides_out,
                                        i_num_bytes_in,
                                        i_num_bytes_out );

  m_xmm_kernel_main = i_kernel_main;
}

void einsum_ir::backend::UnaryLoopsTpp::kernel_main( void const * i_ptr_in,
                                                     void       * io_ptr_out ) {
    libxsmm_meltw_unary_param l_param;
    l_param.in.primary  = (void *) i_ptr_in;
    l_param.out.primary = io_ptr_out;
    m_xmm_kernel_main( &l_param );
}