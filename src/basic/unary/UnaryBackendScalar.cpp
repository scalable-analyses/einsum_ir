
#include "UnaryBackendScalar.h"

template < typename T >
void einsum_ir::basic::UnaryBackendScalar::kernel_zero( void const *,
                                                        void       * o_data ) {
  T * l_data = (T *) o_data;
  *l_data = T(0);
}

template < typename T >
void einsum_ir::basic::UnaryBackendScalar::kernel_relu( void const *,
                                                        void       * io_data ) {
  T * l_data = (T *) io_data;
  *l_data = std::max( *l_data, T(0) );
}

template < typename T >
void einsum_ir::basic::UnaryBackendScalar::kernel_copy( void const * i_data_src,
                                                        void       * io_data_dst ) {
  T const * l_data_src = (T const *) i_data_src;
  T * l_data_dst = (T *) io_data_dst;

  *l_data_dst = *l_data_src;
}


einsum_ir::basic::err_t einsum_ir::basic::UnaryBackendScalar::compile_kernels() {
  //kernel should be of size 1 for scalar interface
  if(    m_m != 1
      || m_n != 1 ) {
    return err_t::COMPILATION_FAILED;
  }

  // determine if all dtypes are FP32 or FP64
  bool l_dtype_all_fp32 = false;
  bool l_dtype_all_fp64 = false;

  if(    m_dtype_in   == FP32
      && m_dtype_comp == FP32
      && m_dtype_out  == FP32 ) {
    l_dtype_all_fp32 = true;
  }
  else if(    m_dtype_in   == FP64
           && m_dtype_comp == FP64
           && m_dtype_out  == FP64 ) {
    l_dtype_all_fp64 = true;
  }
  else {
    return err_t::COMPILATION_FAILED;
  }

  // set main kernel
  if( m_ktype == kernel_t::ZERO ) {
    if( l_dtype_all_fp32 ) {
      m_kernel = &kernel_zero< float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel = &kernel_zero< double >;
    }
  }
  else if( m_ktype == kernel_t::COPY ) {
    if( l_dtype_all_fp32 ) {
      m_kernel = &kernel_copy< float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel = &kernel_copy< double >;
    }
  }
  else if( m_ktype == kernel_t::RELU ) {
    if( l_dtype_all_fp32 ) {
      m_kernel = &kernel_relu< float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel = &kernel_relu< double >;
    }
  }
  else {
    return err_t::COMPILATION_FAILED;
  }

  return err_t::SUCCESS;
}

void einsum_ir::basic::UnaryBackendScalar::kernel_main( void const * i_in,
                                                        void       * io_out ) {

  m_kernel( i_in,
            io_out );
}