#include "ContractionBackendTpp.h"

libxsmm_datatype einsum_ir::binary::ContractionBackendTpp::dtype_to_libxsmm( data_t i_dtype ) {
  if( i_dtype == FP32 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F32;
  }
  else if( i_dtype == FP64 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F64;
  }

  return libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED;
}


void einsum_ir::binary::ContractionBackendTpp::kernel_first_touch( void const * i_out_aux,
                                                                   void       * io_out ){

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


void einsum_ir::binary::ContractionBackendTpp::kernel_last_touch( void const * i_out_aux,
                                                                  void       * io_out ){
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


void einsum_ir::binary::ContractionBackendTpp::kernel_main( void const * i_left,
                                                            void const * i_right,
                                                            void       * io_out ){
  libxsmm_gemm_param l_param;
  l_param.a.primary = (void *) i_left;
  l_param.b.primary = (void *) i_right;
  l_param.c.primary =          io_out;
  l_param.op.tertiary = &m_br;

  m_xmm_kernel_main( &l_param );
}


einsum_ir::err_t einsum_ir::binary::ContractionBackendTpp::compile_kernels(){

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

  // setup bcast 
  libxsmm_bitfield l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_bitfield l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_NONE;

  // column to matrix bcast
  if(         m_stride_m_out_aux >  0
           && m_stride_n_out_aux == 0 ) {
    l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
    l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
  }
  // row to matrix bcast
  else if(    m_stride_m_out_aux == 0
           && m_stride_n_out_aux >  0 ) {
    l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW;
    l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
  }
  // scalar to matrix bcast
  else if(    m_stride_m_out_aux == 0
           && m_stride_n_out_aux == 0 ) {
    l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
    l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
  }

  // first-touch and last-touch shape
  libxsmm_meltw_unary_shape l_shape_single_touch = libxsmm_create_meltw_unary_shape( m_m * m_r,
                                                                                     m_n,
                                                                                     m_ldc,
                                                                                     m_ldc,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out );
  
  libxsmm_meltw_unary_shape l_shape_single_touch_aux_unary = libxsmm_create_meltw_unary_shape( m_m * m_r,
                                                                                               m_n,
                                                                                               m_stride_n_out_aux,
                                                                                               m_ldc,
                                                                                               l_xmm_dtype_out,
                                                                                               l_xmm_dtype_out,
                                                                                               l_xmm_dtype_out );

  libxsmm_meltw_binary_shape l_shape_single_touch_aux_binary = libxsmm_create_meltw_binary_shape( m_m * m_r,
                                                                                                  m_n,
                                                                                                  m_ldc,
                                                                                                  m_stride_n_out_aux,
                                                                                                  m_ldc,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out );

  //first touch kernel
  if( m_ktype_first_touch == kernel_t::ZERO ) {
    m_xmm_kernel_first_touch_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_XOR,
                                                                   l_shape_single_touch,
                                                                   LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }
  else if( m_ktype_first_touch == kernel_t::COPY ) {
    m_xmm_kernel_first_touch_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY,
                                                                   l_shape_single_touch_aux_unary,
                                                                   l_flag_out_aux_unary );
  }
  else if( m_ktype_first_touch == kernel_t::ADD ) {
    m_xmm_kernel_first_touch_binary = libxsmm_dispatch_meltw_binary( LIBXSMM_MELTW_TYPE_BINARY_ADD,
                                                                     l_shape_single_touch_aux_binary,
                                                                     l_flag_out_aux_binary );
  }
  else if( m_ktype_first_touch != kernel_t::UNDEFINED_KTYPE ) {
    return err_t::COMPILATION_FAILED;
  }

  // last touch kernel
  if( m_ktype_last_touch == kernel_t::RELU ) {
    m_xmm_kernel_last_touch_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_RELU,
                                                                  l_shape_single_touch,
                                                                  LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }
  else if( m_ktype_last_touch == kernel_t::ADD ) {
    m_xmm_kernel_last_touch_binary = libxsmm_dispatch_meltw_binary( LIBXSMM_MELTW_TYPE_BINARY_ADD,
                                                                    l_shape_single_touch_aux_binary,
                                                                    l_flag_out_aux_binary );
  }
  else if( m_ktype_last_touch != kernel_t::UNDEFINED_KTYPE ) {
    return err_t::COMPILATION_FAILED;
  }


  //set transpose flags
  libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS('N', 'N');
  l_flags_brgemm |= ( m_trans_a ? LIBXSMM_GEMM_FLAG_TRANS_A : 0);
  l_flags_brgemm |= ( m_trans_b ? LIBXSMM_GEMM_FLAG_TRANS_B : 0);

  //remove packed stride from leading dimensions
  m_lda /= m_r; 
  m_ldb /= m_r; 
  m_ldc /= m_r;

  //create main kernel shape
  libxsmm_gemm_shape l_shape_brgemm;
  l_shape_brgemm = libxsmm_create_gemm_shape( m_m,
                                              m_n,
                                              m_k,
                                              m_lda,
                                              m_ldb,
                                              m_ldc,
                                              l_xmm_dtype_left,
                                              l_xmm_dtype_right,
                                              l_xmm_dtype_out,
                                              l_xmm_dtype_comp );

  //set br type and scale br strides
  libxsmm_gemm_batch_reduce_type l_br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
  if( m_ktype_main == kernel_t::BR_MADD ){
    l_br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    m_br_stride_a *= ce_n_bytes(m_dtype_left);
    m_br_stride_b *= ce_n_bytes(m_dtype_right);
  }

  //set br config
  libxsmm_bitfield l_prefetch_flags_brgemm = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;   
  l_brconfig.br_type = l_br_type;
  l_brconfig.br_stride_a_hint = m_br_stride_a;
  l_brconfig.br_stride_b_hint = m_br_stride_b;
  l_brconfig.br_unroll_hint = 0;

  //create main kernel
  if( m_ktype_main == kernel_t::BR_MADD ||
      m_ktype_main == kernel_t::MADD       ){
    m_xmm_kernel_main = libxsmm_dispatch_brgemm( l_shape_brgemm,
                                                 l_flags_brgemm,
                                                 l_prefetch_flags_brgemm,
                                                 l_brconfig );
  }
  else if( m_ktype_main == kernel_t::PACKED_MADD ){
     m_xmm_kernel_main = libxsmm_create_packed_gemm( l_shape_brgemm,
                                                     l_flags_brgemm,
                                                     l_prefetch_flags_brgemm,
                                                     m_r );
  }

  if( m_xmm_kernel_main == nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  
  return err_t::SUCCESS;
}