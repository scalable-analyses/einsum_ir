
#include "ContractionBackendScalar.h"

template < typename T >
void einsum_ir::binary::ContractionBackendScalar::kernel_zero( void const *,
                                                               void       * o_data ) {
  T * l_data = (T *) o_data;
  *l_data = T(0);
}

template < typename T >
void einsum_ir::binary::ContractionBackendScalar::kernel_relu( void const *,
                                                               void       * io_data ) {
  T * l_data = (T *) io_data;
  *l_data = std::max( *l_data, T(0) );
}

template < typename T >
void einsum_ir::binary::ContractionBackendScalar::kernel_copy( void const * i_data_src,
                                                               void       * io_data_dst ) {
  T const * l_data_src = (T const *) i_data_src;
  T * l_data_dst = (T *) io_data_dst;

  *l_data_dst = *l_data_src;
}

template < typename T_LEFT,
           typename T_RIGHT,
           typename T_OUT >
void einsum_ir::binary::ContractionBackendScalar::kernel_madd( void const * i_left,
                                                               void const * i_right,
                                                               void       * io_out ) {
  T_LEFT  const * l_left  = (T_LEFT  const *) i_left;
  T_RIGHT const * l_right = (T_RIGHT const *) i_right;
  T_OUT         * l_out   = (T_OUT         *) io_out;

  *l_out += (*l_left) * (*l_right);
}

einsum_ir::err_t einsum_ir::binary::ContractionBackendScalar::compile_kernels() {

  //kernel should be of size 1 for scalar interface
  if(    m_m != 1
      || m_n != 1
      || m_k != 1
      || m_r != 1 ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // determine if all dtypes are FP32 or FP64
  bool l_dtype_all_fp32 = false;
  bool l_dtype_all_fp64 = false;

  if(    m_dtype_left  == FP32
      && m_dtype_right == FP32
      && m_dtype_comp  == FP32
      && m_dtype_out   == FP32 ) {
    l_dtype_all_fp32 = true;
  }
  else if(    m_dtype_left  == FP64
           && m_dtype_right == FP64
           && m_dtype_comp  == FP64
           && m_dtype_out   == FP64 ) {
    l_dtype_all_fp64 = true;
  }

  // first-touch kernel
  if( m_ktype_first_touch == kernel_t::ZERO ) {
    if( l_dtype_all_fp32 ) {
      m_kernel_first_touch = &kernel_zero< float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel_first_touch = &kernel_zero< double >;
    }
    else {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  else if( m_ktype_first_touch == kernel_t::COPY ) {
    if( l_dtype_all_fp32 ) {
      m_kernel_first_touch = &kernel_copy< float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel_first_touch = &kernel_copy< double >;
    }
    else {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  else if( m_ktype_first_touch != kernel_t::UNDEFINED_KTYPE ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // main kernel
  if( m_ktype_main == kernel_t::MADD ) {
    if( l_dtype_all_fp32 ) {
      m_kernel_main = &kernel_madd< float, float, float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel_main = &kernel_madd< double, double, double >;
    }
    else {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  else {
    return einsum_ir::COMPILATION_FAILED;
  }

  // last-touch kernel
  if( m_ktype_last_touch == kernel_t::RELU ) {
    if( l_dtype_all_fp32 ) {
      m_kernel_last_touch = &kernel_relu< float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel_last_touch = &kernel_relu< double >;
    }
    else {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  else if( m_ktype_last_touch != UNDEFINED_KTYPE ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  return einsum_ir::SUCCESS;
}


void einsum_ir::binary::ContractionBackendScalar::kernel_first_touch( void const * i_out_aux,
                                                                      void       * io_out ) {
  if( m_kernel_first_touch != nullptr ) {
    m_kernel_first_touch( i_out_aux,
                          io_out );
  }
}

void einsum_ir::binary::ContractionBackendScalar::kernel_main( void const * i_left,
                                                               void const * i_right,
                                                               void       * io_out ) {
  m_kernel_main( i_left,
                 i_right,
                 io_out );
}

void einsum_ir::binary::ContractionBackendScalar::kernel_last_touch( void const * i_out_aux,
                                                                     void       * io_out ) {
  if( m_kernel_last_touch != nullptr ) {
    m_kernel_last_touch( i_out_aux,
                         io_out );
  }
}