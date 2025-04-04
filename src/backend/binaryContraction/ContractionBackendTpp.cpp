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
  libxsmm_datatype l_xmm_dtype_left  = dtype_to_libxsmm( m_dtype_left  );
  libxsmm_datatype l_xmm_dtype_right = dtype_to_libxsmm( m_dtype_right );
  libxsmm_datatype l_xmm_dtype_out   = dtype_to_libxsmm( m_dtype_out   );
  libxsmm_datatype l_xmm_dtype_comp  = dtype_to_libxsmm( m_dtype_comp );

  if(    l_xmm_dtype_left  == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED
      || l_xmm_dtype_right == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED
      || l_xmm_dtype_comp  == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED
      || l_xmm_dtype_out   == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  //search for sizes and stides and update gemm
  libxsmm_gemm_batch_reduce_type l_br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
  int64_t l_loop_id = m_loop_sizes.size() - 1;

  //get K size
  libxsmm_blasint l_k = 1;
  libxsmm_blasint l_lda = m_loop_strides_left[l_loop_id];
  while( m_loop_dim_type[ l_loop_id] == dim_t::K    && 
         m_loop_exec_type[l_loop_id] == exec_t::PRIM && 
         l_loop_id >= 0 ){
    l_k *= m_loop_sizes[l_loop_id];
    l_loop_id--;
  }
  //get n size
  libxsmm_blasint l_n = 1;
  libxsmm_blasint l_ldb = m_loop_strides_right[l_loop_id];
  libxsmm_blasint l_ldc = m_loop_strides_out[l_loop_id];
  while( m_loop_dim_type[ l_loop_id] == dim_t::N    && 
         m_loop_exec_type[l_loop_id] == exec_t::PRIM && 
         l_loop_id >= 0 ){
    l_n *= m_loop_sizes[l_loop_id];
    l_loop_id--;
  }
  //get m size
  libxsmm_blasint l_m = 1;
  while( m_loop_dim_type[ l_loop_id] == dim_t::M    && 
         m_loop_exec_type[l_loop_id] == exec_t::PRIM && 
         l_loop_id >= 0 ){
    l_m *= m_loop_sizes[l_loop_id];
    l_loop_id--;
  }
  //get br size
  m_br = 1;
  libxsmm_blasint l_br_stride_a = 0;
  libxsmm_blasint l_br_stride_b = 0;
  if( m_ktype_main == kernel_t::BR_MADD ){
    l_br_stride_a = m_loop_strides_left[l_loop_id] * ce_n_bytes(m_dtype_left);
    l_br_stride_b = m_loop_strides_right[l_loop_id] * ce_n_bytes(m_dtype_right);
    while( m_loop_dim_type[ l_loop_id] == dim_t::K    && 
           m_loop_exec_type[l_loop_id] == exec_t::PRIM && 
           l_loop_id >= 0 ){
      m_br *= m_loop_sizes[l_loop_id];
      l_loop_id--;
    }
    l_br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  }

  //std::cout << l_m << " " << l_n << " " << l_ldc << std::endl;

  // first-touch and last-touch shape
  libxsmm_meltw_unary_shape l_shape_single_touch = libxsmm_create_meltw_unary_shape( l_m,
                                                                                     l_n,
                                                                                     l_ldc,
                                                                                     l_ldc,
                                                                                     libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                                                                     libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                                                                     libxsmm_datatype::LIBXSMM_DATATYPE_F32 );

  //create first touch kernel
  if( m_ktype_first_touch == kernel_t::ZERO ) {
    m_xmm_kernel_first_touch_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_XOR,
                                                                   l_shape_single_touch,
                                                                   LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }

  //TODO create other first and last touch kernels

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
  l_brconfig.br_type = l_br_type;
  l_brconfig.br_stride_a_hint = l_br_stride_a;
  l_brconfig.br_stride_b_hint = l_br_stride_b;
  l_brconfig.br_unroll_hint = 0;

  m_xmm_kernel_main = libxsmm_dispatch_brgemm( l_shape_brgemm,
                                               l_flags_brgemm,
                                               l_prefetch_flags_brgemm,
                                               l_brconfig );

  if( m_xmm_kernel_main == nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  
  return err_t::SUCCESS;
}