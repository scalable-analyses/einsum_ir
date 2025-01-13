#include "ContractionBackendTpp.h"
#include <iostream>

libxsmm_datatype einsum_ir::backend::ContractionBackendTpp::dtype_to_libxsmm( data_t i_dtype ) {
  if( i_dtype == FP32 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F32;
  }
  else if( i_dtype == FP64 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F64;
  }

  return libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED;
}


void einsum_ir::backend::ContractionBackendTpp::kernel_first_touch( void const * i_out_aux,
                                                                    void       * io_out ){
  libxsmm_meltw_unary_param l_param;
  l_param.in.primary  = (void *) i_out_aux;
  l_param.out.primary =          io_out;
  m_xmm_kernel_first_touch_unary( &l_param );
}


void einsum_ir::backend::ContractionBackendTpp::kernel_last_touch( void const * i_out_aux,
                                                                   void       * io_out ){

}


void einsum_ir::backend::ContractionBackendTpp::kernel_main( void const * i_left,
                                                             void const * i_right,
                                                             void       * io_out ){
  libxsmm_gemm_param l_param;
  l_param.a.primary = (void *) i_left;
  l_param.b.primary = (void *) i_right;
  l_param.c.primary =          io_out;
  if( m_ktype_main == kernel_t::BR_MADD ){
    l_param.op.tertiary = &m_br;
  }

  m_xmm_kernel_main( &l_param );
}


einsum_ir::err_t einsum_ir::backend::ContractionBackendTpp::compile_kernels(){

  // libxsmm data types
  libxsmm_datatype l_xmm_dtype_left  = dtype_to_libxsmm( m_dtype_left );
  libxsmm_datatype l_xmm_dtype_right = dtype_to_libxsmm( m_dtype_right );
  libxsmm_datatype l_xmm_dtype_comp  = dtype_to_libxsmm( m_dtype_comp );
  libxsmm_datatype l_xmm_dtype_out   = dtype_to_libxsmm( m_dtype_out );

  if(    l_xmm_dtype_left  == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED
      || l_xmm_dtype_right == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED
      || l_xmm_dtype_comp  == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED
      || l_xmm_dtype_out   == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  int64_t l_num_loops = m_loop_sizes.size();
  libxsmm_blasint l_m = m_loop_sizes[l_num_loops-3];
  libxsmm_blasint l_n = m_loop_sizes[l_num_loops-2];
  libxsmm_blasint l_k = m_loop_sizes[l_num_loops-1];

  libxsmm_blasint l_lda = m_loop_strides_left[ l_num_loops-1];
  libxsmm_blasint l_ldb = m_loop_strides_right[l_num_loops-2];
  libxsmm_blasint l_ldc = m_loop_strides_out[  l_num_loops-2];

  std::cout << l_m << " " << l_n <<" " << l_k << std::endl; 
  std::cout << l_lda << " " << l_ldb <<" " << l_ldc << std::endl; 

  // first-touch and last-touch shape
  libxsmm_meltw_unary_shape l_shape_single_touch = libxsmm_create_meltw_unary_shape( l_m,
                                                                                     l_n,
                                                                                     l_ldc,
                                                                                     l_ldc,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out );

  //create first touch kernel
  if( m_ktype_first_touch == kernel_t::ZERO ) {
    m_xmm_kernel_first_touch_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_XOR,
                                                                   l_shape_single_touch,
                                                                   LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }


  // create main kernel
  libxsmm_gemm_shape l_shape_brgemm;
  libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags_brgemm = 0;

  l_shape_brgemm = libxsmm_create_gemm_shape( l_m,
                                              l_n,
                                              l_k,
                                              l_lda,
                                              l_ldb,
                                              l_ldc,
                                              l_xmm_dtype_left,
                                              l_xmm_dtype_right,
                                              l_xmm_dtype_out,
                                              l_xmm_dtype_comp );

  libxsmm_gemm_batch_reduce_config l_brconfig;
  if( m_ktype_main == kernel_t::BR_MADD ){
    m_br = m_loop_sizes[l_num_loops-4]; //can't use class variable because of poor performance. False sharing?
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = m_loop_strides_left[ l_num_loops-4] * ce_n_bytes(m_dtype_left);
    l_brconfig.br_stride_b_hint = m_loop_strides_right[l_num_loops-4] * ce_n_bytes(m_dtype_right);
    l_brconfig.br_unroll_hint = 0; //TODO could remove unroll hint to improve performance
  }
  else {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = 0;
  }
  


  m_xmm_kernel_main = libxsmm_dispatch_brgemm( l_shape_brgemm,
                                               l_flags_brgemm,
                                               l_prefetch_flags_brgemm,
                                               l_brconfig );

  if( m_xmm_kernel_main == nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  return err_t::SUCCESS;
}