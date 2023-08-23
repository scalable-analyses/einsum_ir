#include "ContractionLoopsTpp.h"

void einsum_ir::backend::ContractionLoopsTpp::init( int64_t                             i_num_dims_c,
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
                                                    int64_t                     const * i_strides_out_c,
                                                    int64_t                     const * i_strides_out_m,
                                                    int64_t                     const * i_strides_out_n,
                                                    int64_t                             i_num_bytes_scalar_left,
                                                    int64_t                             i_num_bytes_scalar_right,
                                                    int64_t                             i_num_bytes_scalar_out,
                                                    libxsmm_meltwfunction_unary const   i_xmm_kernel_first_touch,
                                                    libxsmm_gemmfunction        const   i_xmm_kernel_inner,
                                                    libxsmm_meltwfunction_unary const   i_xmm_kernel_last_touch ) {
                                                                
  ContractionLoops::init( i_num_dims_c,
                          i_num_dims_m,
                          i_num_dims_n,
                          i_num_dims_k,
                          i_sizes_c,
                          i_sizes_m,
                          i_sizes_n,
                          i_sizes_k,
                          i_strides_in_left_c,
                          i_strides_in_left_m,
                          i_strides_in_left_k,
                          i_strides_in_right_c,
                          i_strides_in_right_n,
                          i_strides_in_right_k,
                          i_strides_out_c,
                          i_strides_out_m,
                          i_strides_out_n,
                          i_num_bytes_scalar_left,
                          i_num_bytes_scalar_right,
                          i_num_bytes_scalar_out );

  m_xmm_kernel_first_touch = i_xmm_kernel_first_touch;
  m_xmm_kernel_inner       = i_xmm_kernel_inner;
  m_xmm_kernel_last_touch  = i_xmm_kernel_last_touch;
}

void einsum_ir::backend::ContractionLoopsTpp::kernel_first_touch( void * io_out ) {
  if( m_xmm_kernel_first_touch != nullptr ) {
    libxsmm_meltw_unary_param l_param;
    l_param.in.primary = io_out;
    l_param.out.primary = io_out;
    m_xmm_kernel_first_touch( &l_param );
  }
}

void einsum_ir::backend::ContractionLoopsTpp::kernel_inner( void const * i_left,
                                                             void const * i_right,
                                                             void       * io_out ) {
  libxsmm_gemm_param l_param;
  l_param.a.primary = (void *) i_left;
  l_param.b.primary = (void *) i_right;
  l_param.c.primary =          io_out;

  m_xmm_kernel_inner( &l_param );
}

void einsum_ir::backend::ContractionLoopsTpp::kernel_last_touch( void * io_out ) {
  if( m_xmm_kernel_last_touch != nullptr ) {
    libxsmm_meltw_unary_param l_param;
    l_param.in.primary = io_out;
    l_param.out.primary = io_out;
    m_xmm_kernel_last_touch( &l_param );
  }
}