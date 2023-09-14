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

einsum_ir::err_t einsum_ir::backend::BinaryContractionScalar::compile( tenord_t i_tensor_ordering ) {
  BinaryContraction::compile_base();

  // reorder input dimensions if requested
  int64_t const * l_dim_ids_left  = m_dim_ids_left_native;
  int64_t const * l_dim_ids_right = m_dim_ids_right_native;
  m_dim_ids_left_ordered.resize(0);
  m_dim_ids_right_ordered.resize(0);

  if( i_tensor_ordering == LEFT_NATIVE_RIGHT_NATIVE_OUT_NATIVE ) {}
  else if( i_tensor_ordering == LEFT_BC_BM_BI_BK_RIGHT_BC_BN_BJ_BK_OUT_NATIVE ) {
    m_dim_ids_left_ordered.resize( m_num_dims_left );
    m_dim_ids_right_ordered.resize( m_num_dims_right );
    l_dim_ids_left = m_dim_ids_left_ordered.data();
    l_dim_ids_right = m_dim_ids_right_ordered.data();

    order_dims_in( i_tensor_ordering,
                   m_num_dims_c,
                   m_num_dims_m,
                   m_num_dims_n,
                   m_num_dims_k,
                   m_num_dims_i,
                   m_num_dims_j,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   m_dim_ids_c.data(),
                   m_dim_ids_m.data(),
                   m_dim_ids_n.data(),
                   m_dim_ids_k.data(),
                   m_dim_ids_i.data(),
                   m_dim_ids_j.data(),
                   m_dim_ids_left_ordered.data(),
                   m_dim_ids_right_ordered.data() );

    for( int64_t l_le = 0; l_le < m_num_dims_left; l_le++ ) {
      if( m_dim_ids_left_native[l_le] != m_dim_ids_left_ordered[l_le] ) {
        m_tensors_in_reordered = true;
        break;
      }
    }
    for( int64_t l_ri = 0; l_ri < m_num_dims_right; l_ri++ ) {
      if( m_dim_ids_right_native[l_ri] != m_dim_ids_right_ordered[l_ri] ) {
        m_tensors_in_reordered = true;
        break;
      }
    }
  }
  else {
    return COMPILATION_FAILED;
  }

  // derive strides
  m_strides_left_c.resize( m_num_dims_c );
  m_strides_left_m.resize( m_num_dims_m );
  m_strides_left_k.resize( m_num_dims_k );
  m_strides_left_i.resize( m_num_dims_i );

  m_strides_right_c.resize( m_num_dims_c );
  m_strides_right_n.resize( m_num_dims_n );
  m_strides_right_k.resize( m_num_dims_k );
  m_strides_right_j.resize( m_num_dims_j );

  m_strides_out_aux_c.resize( m_num_dims_c );
  m_strides_out_aux_m.resize( m_num_dims_m );
  m_strides_out_aux_n.resize( m_num_dims_n );

  m_strides_out_c.resize( m_num_dims_c );
  m_strides_out_m.resize( m_num_dims_m );
  m_strides_out_n.resize( m_num_dims_n );

  strides( m_num_dims_left,
           m_num_dims_right,
           m_num_dims_out,
           m_num_dims_c,
           m_num_dims_m,
           m_num_dims_n,
           m_num_dims_k,
           m_num_dims_i,
           m_num_dims_j,
           l_dim_ids_left,
           l_dim_ids_right,
           m_dim_ids_out,
           m_dim_ids_c.data(),
           m_dim_ids_m.data(),
           m_dim_ids_n.data(),
           m_dim_ids_k.data(),
           m_dim_ids_i.data(),
           m_dim_ids_j.data(),
           *m_dim_sizes_outer_left,
           *m_dim_sizes_outer_right,
           *m_dim_sizes_outer_out_aux,
           *m_dim_sizes_outer_out,
           m_strides_left_c.data(),
           m_strides_left_m.data(),
           m_strides_left_k.data(),
           m_strides_left_i.data(),
           m_strides_right_c.data(),
           m_strides_right_n.data(),
           m_strides_right_k.data(),
           m_strides_right_j.data(),
           m_strides_out_aux_c.data(),
           m_strides_out_aux_m.data(),
           m_strides_out_aux_n.data(),
           m_strides_out_c.data(),
           m_strides_out_m.data(),
           m_strides_out_n.data() );

  // treat secondary I dimensions as K internally
  for( int64_t l_di_i = 0; l_di_i < m_num_dims_i; l_di_i++ ) {
    int64_t l_di_n = 0;
    err_t l_err = link_secondary_to_primary( m_dim_ids_i[l_di_i],
                                             m_num_dims_n,
                                             m_dim_ids_n.data(),
                                             *m_dim_link_s_to_p,
                                             l_di_n );
    if( l_err != err_t::SUCCESS ) {
      return err_t::COMPILATION_FAILED;
    }

    // corresponding contraction info
    int64_t l_size_s   = m_sizes_i[l_di_i];
    int64_t l_stride_p = m_strides_right_n[l_di_n];
    int64_t l_stride_s = m_strides_left_i[l_di_i];

    // add to K dimensions
    m_num_dims_k++;
    m_sizes_k.push_back( l_size_s );
    m_strides_left_k.push_back( l_stride_s );
    m_strides_right_k.push_back( l_stride_p );
  }

  // treat secondary J dimensions as K internally
  for( int64_t l_di_j = 0; l_di_j < m_num_dims_j; l_di_j++ ) {
    int64_t l_di_m = 0;
    err_t l_err = link_secondary_to_primary( m_dim_ids_j[l_di_j],
                                             m_num_dims_m,
                                             m_dim_ids_m.data(),
                                             *m_dim_link_s_to_p,
                                             l_di_m );
    if( l_err != err_t::SUCCESS ) {
      return err_t::COMPILATION_FAILED;
    }

    // corresponding contraction info
    int64_t l_size_s   = m_sizes_j[l_di_j];
    int64_t l_stride_p = m_strides_left_m[l_di_m];
    int64_t l_stride_s = m_strides_right_j[l_di_j];

    // add to K dimensions
    m_num_dims_k++;
    m_sizes_k.push_back( l_size_s );
    m_strides_left_k.push_back( l_stride_p );
    m_strides_right_k.push_back( l_stride_s );
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

  m_cont_loops.init( m_num_dims_c,
                     m_num_dims_m,
                     m_num_dims_n,
                     m_num_dims_k,
                     m_sizes_c.data(),
                     m_sizes_m.data(),
                     m_sizes_n.data(),
                     m_sizes_k.data(),
                     m_strides_left_c.data(),
                     m_strides_left_m.data(),
                     m_strides_left_k.data(),
                     m_strides_right_c.data(),
                     m_strides_right_n.data(),
                     m_strides_right_k.data(),
                     m_strides_out_aux_c.data(),
                     m_strides_out_aux_m.data(),
                     m_strides_out_aux_n.data(),
                     m_strides_out_c.data(),
                     m_strides_out_m.data(),
                     m_strides_out_n.data(),
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

einsum_ir::err_t einsum_ir::backend::BinaryContractionScalar::compile() {
  err_t l_err = compile( LEFT_NATIVE_RIGHT_NATIVE_OUT_NATIVE );
  return l_err;
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