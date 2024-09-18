#include "ContractionLoopsTpp.h"

void einsum_ir::backend::ContractionLoopsTpp::init( int64_t                              i_num_dims_c,
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
                                                    std::vector<int64_t>               * i_loop_ids,
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
                                                    ContractionPackingTpp              * i_packing ) {
  ContractionLoops::init( i_num_dims_c,
                          i_num_dims_m,
                          i_num_dims_n,
                          i_num_dims_k,
                          i_dim_ids_c,
                          i_dim_ids_m,
                          i_dim_ids_n,
                          i_dim_ids_k,
                          i_sizes,
                          i_strides_left,
                          i_strides_right,
                          i_strides_out_aux,
                          i_strides_out,
                          i_dim_type,
                          i_loop_ids,
                          i_num_bytes_scalar_left,
                          i_num_bytes_scalar_right,
                          i_num_bytes_scalar_out,
                          i_ktype_first_touch,
                          i_ktype_main,
                          i_ktype_last_touch,
                          i_packing );

  m_xmm_kernel_first_touch_unary  = i_xmm_kernel_first_touch_unary;
  m_xmm_kernel_first_touch_binary = i_xmm_kernel_first_touch_binary;
  m_xmm_kernel_main               = i_xmm_kernel_main;
  m_xmm_kernel_last_touch_unary   = i_xmm_kernel_last_touch_unary;
  m_xmm_kernel_last_touch_binary  = i_xmm_kernel_last_touch_binary;
}

void einsum_ir::backend::ContractionLoopsTpp::kernel_first_touch( void const * i_out_aux,
                                                                  void       * io_out ) {
  if( m_xmm_kernel_first_touch_unary != nullptr ) {
    libxsmm_meltw_unary_param l_param;
    l_param.in.primary  = (void *) i_out_aux;
    l_param.out.primary =          io_out;
    m_xmm_kernel_first_touch_unary( &l_param );
  }
  else if( m_xmm_kernel_first_touch_binary != nullptr ) {
    libxsmm_meltw_binary_param l_param;
    l_param.in0.primary = (void *) io_out;
    l_param.in1.primary = (void *) i_out_aux;
    l_param.out.primary =          io_out;
    m_xmm_kernel_first_touch_binary( &l_param );
  }
}

void einsum_ir::backend::ContractionLoopsTpp::kernel_main( void const * i_left,
                                                           void const * i_right,
                                                           void       * io_out ) {
  libxsmm_gemm_param l_param;
  l_param.a.primary = (void *) i_left;
  l_param.b.primary = (void *) i_right;
  l_param.c.primary =          io_out;

  m_xmm_kernel_main( &l_param );
}

void einsum_ir::backend::ContractionLoopsTpp::kernel_last_touch( void const * i_out_aux,
                                                                 void       * io_out ) {
  if( m_xmm_kernel_last_touch_unary != nullptr ) {
    libxsmm_meltw_unary_param l_param;
    l_param.in.primary = io_out;
    l_param.out.primary = io_out;
    m_xmm_kernel_last_touch_unary( &l_param );
  }
  else if( m_xmm_kernel_last_touch_binary != nullptr ) {
    libxsmm_meltw_binary_param l_param;
    l_param.in0.primary = (void *) io_out;
    l_param.in1.primary = (void *) i_out_aux;
    l_param.out.primary =          io_out;
    m_xmm_kernel_last_touch_binary( &l_param );
  }
}