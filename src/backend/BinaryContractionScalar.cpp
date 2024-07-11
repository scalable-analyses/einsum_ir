#include "BinaryContractionScalar.h"
#include "ContractionLoopsSimple.h"
#include <cmath>

template < typename T >
void einsum_ir::backend::BinaryContractionScalar::kernel_zero( void const *,
                                                               void       * o_data ) {
  T * l_data = (T *) o_data;
  *l_data = T(0);
}

template < typename T >
void einsum_ir::backend::BinaryContractionScalar::kernel_relu( void const *,
                                                               void       * io_data ) {
  T * l_data = (T *) io_data;
  *l_data = std::max( *l_data, T(0) );
}

template < typename T >
void einsum_ir::backend::BinaryContractionScalar::kernel_copy( void const * i_data_src,
                                                               void       * io_data_dst ) {
  T const * l_data_src = (T const *) i_data_src;
  T * l_data_dst = (T *) io_data_dst;

  *l_data_dst = *l_data_src;
}

template < typename T_LEFT,
           typename T_RIGHT,
           typename T_OUT >
void einsum_ir::backend::BinaryContractionScalar::kernel_madd( void const * i_left,
                                                               void const * i_right,
                                                               void       * io_out ) {
  T_LEFT  const * l_left  = (T_LEFT  const *) i_left;
  T_RIGHT const * l_right = (T_RIGHT const *) i_right;
  T_OUT         * l_out   = (T_OUT         *) io_out;

  *l_out += (*l_left) * (*l_right);
}

einsum_ir::err_t einsum_ir::backend::BinaryContractionScalar::compile() {
  BinaryContraction::compile_base();

  // derive strides
  std::map< int64_t, int64_t > l_strides_left;
  std::map< int64_t, int64_t > l_strides_right;
  std::map< int64_t, int64_t > l_strides_out;
  std::map< int64_t, int64_t > l_strides_out_aux;

  strides( m_num_dims_left,
           m_dim_ids_left,
           m_dim_sizes_outer_left,
           &l_strides_left );

  strides( m_num_dims_right,
           m_dim_ids_right,
           m_dim_sizes_outer_right,
           &l_strides_right );

  strides( m_num_dims_out,
           m_dim_ids_out,
           m_dim_sizes_outer_out,
           &l_strides_out );

  if( m_dim_sizes_outer_out_aux != nullptr ) {
    strides( m_num_dims_out,
             m_dim_ids_out,
             m_dim_sizes_outer_out_aux,
             &l_strides_out_aux );
  }
  else {
    l_strides_out_aux = l_strides_out;
  }

  // derive stride of non-blocked dimensions
  for( std::size_t l_di = 0; l_di < m_dim_ids_c.size(); l_di++ ) {
    int64_t l_dim_id = m_dim_ids_c[l_di];
    m_strides_left_c.push_back(    l_strides_left[    l_dim_id ] );
    m_strides_right_c.push_back(   l_strides_right[   l_dim_id ] );
    m_strides_out_c.push_back(     l_strides_out[     l_dim_id ] );
    m_strides_out_aux_c.push_back( l_strides_out_aux[ l_dim_id ] );
  }

  for( std::size_t l_di = 0; l_di < m_dim_ids_m.size(); l_di++ ) {
    int64_t l_dim_id = m_dim_ids_m[l_di];
    m_strides_left_m.push_back(    l_strides_left[    l_dim_id ] );
    m_strides_out_m.push_back(     l_strides_out[     l_dim_id ] );
    m_strides_out_aux_m.push_back( l_strides_out_aux[ l_dim_id ] );
  }

  for( std::size_t l_di = 0; l_di < m_dim_ids_n.size(); l_di++ ) {
    int64_t l_dim_id = m_dim_ids_n[l_di];
    m_strides_right_n.push_back(   l_strides_right[   l_dim_id ] );
    m_strides_out_n.push_back(     l_strides_out[     l_dim_id ] );
    m_strides_out_aux_n.push_back( l_strides_out_aux[ l_dim_id ] );
  }

  for( std::size_t l_di = 0; l_di < m_dim_ids_k.size(); l_di++ ) {
    int64_t l_dim_id = m_dim_ids_k[l_di];
    m_strides_left_k.push_back(  l_strides_left[  l_dim_id ] );
    m_strides_right_k.push_back( l_strides_right[ l_dim_id ] );
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
  if( m_ktype_first_touch == ZERO ) {
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
  else if( m_ktype_first_touch == COPY ) {
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
  else if( m_ktype_first_touch != UNDEFINED_KTYPE ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // main kernel
  if( m_ktype_main == MADD ) {
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
  if( m_ktype_last_touch == RELU ) {
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

  m_cont_loops.init( m_dim_ids_c.size(),
                     m_dim_ids_m.size(),
                     m_dim_ids_n.size(),
                     m_dim_ids_k.size(),
                     m_dim_ids_c.data(),
                     m_dim_ids_m.data(),
                     m_dim_ids_n.data(),
                     m_dim_ids_k.data(),
                     m_dim_sizes_inner,
                     &l_strides_left,
                     &l_strides_right,
                     &l_strides_out_aux,
                     &l_strides_out,
                     &m_dim_types,
                     ce_n_bytes( m_dtype_left ),
                     ce_n_bytes( m_dtype_right ),
                     ce_n_bytes( m_dtype_out ),
                     m_kernel_first_touch,
                     m_kernel_main,
                     m_kernel_last_touch );

  err_t l_err = m_cont_loops.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return einsum_ir::COMPILATION_FAILED;
  }


  m_compiled = true;

  return einsum_ir::SUCCESS;
}

void einsum_ir::backend::BinaryContractionScalar::contract( void const * i_tensor_left,
                                                            void const * i_tensor_right,
                                                            void const * i_tensor_out_aux,
                                                            void       * io_tensor_out ) {
  m_cont_loops.contract( i_tensor_left,
                         i_tensor_right,
                         i_tensor_out_aux,
                         io_tensor_out );
}

void einsum_ir::backend::BinaryContractionScalar::contract( void const * i_tensor_left,
                                                            void const * i_tensor_right,
                                                            void       * io_tensor_out ) {
  contract( i_tensor_left,
            i_tensor_right,
            nullptr,
            io_tensor_out );
}