#include "UnaryBackendTpp.h"

libxsmm_datatype einsum_ir::basic::UnaryBackendTpp::dtype_to_libxsmm( data_t i_dtype ) {
  if( i_dtype == FP32 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F32;
  }
  else if( i_dtype == FP64 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F64;
  }

  return libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED;
}

void einsum_ir::basic::UnaryBackendTpp::kernel_main( void const * i_out_aux,
                                                     void       * io_out ){
  if( m_xmm_kernel_unary != nullptr ) {
    libxsmm_meltw_unary_param l_param;
    l_param.in.primary  = (void *) i_out_aux;
    l_param.out.primary =          io_out;
    m_xmm_kernel_unary( &l_param );
  }
  else if( m_xmm_kernel_binary != nullptr ) {
    libxsmm_meltw_binary_param l_param;
    l_param.in0.primary = (void *) io_out;
    l_param.in1.primary = (void *) i_out_aux;
    l_param.out.primary =          io_out;
    m_xmm_kernel_binary( &l_param );
  }
}

einsum_ir::basic::err_t einsum_ir::basic::UnaryBackendTpp::compile_kernels(){

  // libxsmm data types
  libxsmm_datatype l_xmm_dtype_in   = dtype_to_libxsmm( m_dtype_in   );
  libxsmm_datatype l_xmm_dtype_out  = dtype_to_libxsmm( m_dtype_out  );
  libxsmm_datatype l_xmm_dtype_comp = dtype_to_libxsmm( m_dtype_comp );

  if(    l_xmm_dtype_in   == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED
      || l_xmm_dtype_comp == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED
      || l_xmm_dtype_out  == libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED ) {
    return err_t::COMPILATION_FAILED;
  }

  // first-touch and last-touch shape
  libxsmm_meltw_unary_shape l_shape_single_touch = libxsmm_create_meltw_unary_shape( m_m,
                                                                                     m_n,
                                                                                     m_ldb,
                                                                                     m_ldb,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out );
  
  libxsmm_meltw_unary_shape l_shape_single_touch_aux_unary = libxsmm_create_meltw_unary_shape( m_m,
                                                                                               m_n,
                                                                                               m_lda,
                                                                                               m_ldb,
                                                                                               l_xmm_dtype_out,
                                                                                               l_xmm_dtype_out,
                                                                                               l_xmm_dtype_out );

  libxsmm_meltw_binary_shape l_shape_single_touch_aux_binary = libxsmm_create_meltw_binary_shape( m_m,
                                                                                                  m_n,
                                                                                                  m_ldb,
                                                                                                  m_lda,
                                                                                                  m_ldb,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out );

  //first touch kernel
  if( m_ktype == kernel_t::ZERO ) {
    m_xmm_kernel_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_XOR,
                                                       l_shape_single_touch,
                                                       LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }
  else if( m_ktype == kernel_t::COPY ) {
    if(m_trans_a){
      m_xmm_kernel_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT,
                                                        l_shape_single_touch_aux_unary,
                                                        LIBXSMM_MELTW_FLAG_UNARY_NONE );
    }
    else{
      m_xmm_kernel_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY,
                                                        l_shape_single_touch_aux_unary,
                                                        LIBXSMM_MELTW_FLAG_UNARY_NONE );
    }
  }
  else if( m_ktype == kernel_t::ADD ) {
    m_xmm_kernel_binary = libxsmm_dispatch_meltw_binary( LIBXSMM_MELTW_TYPE_BINARY_ADD,
                                                         l_shape_single_touch_aux_binary,
                                                         LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }
  else {
    return err_t::COMPILATION_FAILED;
  }

  if(m_xmm_kernel_unary == nullptr && m_xmm_kernel_binary == nullptr) {
    return err_t::COMPILATION_FAILED;
  }

  return err_t::SUCCESS;
}