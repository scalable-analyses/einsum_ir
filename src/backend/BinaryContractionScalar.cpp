#include "BinaryContractionScalar.h"
#include "ContractionLoopsSimple.h"
#include <cmath>

template < typename T >
void einsum_ir::backend::BinaryContractionScalar::kernel_zero( void * o_data ) {
  T * l_data = (T *) o_data;
  *l_data = T(0);
}

template < typename T >
void einsum_ir::backend::BinaryContractionScalar::kernel_relu( void * io_data ) {
  T * l_data = (T *) io_data;
  *l_data = std::max( *l_data, T(0) );
}

template < typename T_LEFT,
           typename T_RIGHT,
           typename T_OUT >
void einsum_ir::backend::BinaryContractionScalar::kernel_madd( void const * i_in_left,
                                                               void const * i_in_right,
                                                               void       * io_out ) {
  T_LEFT  const * l_in_left  = (T_LEFT  const *) i_in_left;
  T_RIGHT const * l_in_right = (T_RIGHT const *) i_in_right;
  T_OUT         * l_out      = (T_OUT         *) io_out;

  *l_out += (*l_in_left) * (*l_in_right);
}

einsum_ir::err_t einsum_ir::backend::BinaryContractionScalar::compile( tenord_t i_tensor_ordering ) {
  BinaryContraction::compile_base();

  // reorder input dimensions if requested
  int64_t const * l_dim_ids_in_left  = m_dim_ids_in_left_native;
  int64_t const * l_dim_ids_in_right = m_dim_ids_in_right_native;
  m_dim_ids_left_ordered.resize(0);
  m_dim_ids_right_ordered.resize(0);

  if( i_tensor_ordering == LEFT_NATIVE_RIGHT_NATIVE_OUT_NATIVE ) {}
  else if( i_tensor_ordering == LEFT_BC_BM_BK_RIGHT_BC_BN_BK_OUT_NATIVE ) {
    m_dim_ids_left_ordered.resize( m_num_dims_in_left );
    m_dim_ids_right_ordered.resize( m_num_dims_in_right );
    l_dim_ids_in_left = m_dim_ids_left_ordered.data();
    l_dim_ids_in_right = m_dim_ids_right_ordered.data();

    order_dims_in( i_tensor_ordering,
                   m_num_dims_c,
                   m_num_dims_m,
                   m_num_dims_n,
                   m_num_dims_k,
                   0,
                   0,
                   0,
                   0,
                   m_dim_ids_c.data(),
                   m_dim_ids_m.data(),
                   m_dim_ids_n.data(),
                   m_dim_ids_k.data(),
                   m_dim_ids_left_ordered.data(),
                   m_dim_ids_right_ordered.data() );

    for( int64_t l_le = 0; l_le < m_num_dims_in_left; l_le++ ) {
      if( m_dim_ids_in_left_native[l_le] != m_dim_ids_left_ordered[l_le] ) {
        m_tensors_in_reordered = true;
        break;
      }
    }
    for( int64_t l_ri = 0; l_ri < m_num_dims_in_right; l_ri++ ) {
      if( m_dim_ids_in_right_native[l_ri] != m_dim_ids_right_ordered[l_ri] ) {
        m_tensors_in_reordered = true;
        break;
      }
    }
  }
  else {
    return COMPILATION_FAILED;
  }

  // derive strides
  std::map< int64_t, int64_t > l_map_id_stride_in_left;
  std::map< int64_t, int64_t > l_map_id_stride_in_right;
  std::map< int64_t, int64_t > l_map_id_stride_out;

  strides( m_num_dims_in_left,
           l_dim_ids_in_left,
           *m_dim_sizes,
           l_map_id_stride_in_left );

  strides( m_num_dims_in_right,
           l_dim_ids_in_right,
           *m_dim_sizes,
           l_map_id_stride_in_right );

  strides( m_num_dims_out,
           m_dim_ids_out,
           *m_dim_sizes,
           l_map_id_stride_out );

  m_strides_in_left_c.resize( m_num_dims_c );
  m_strides_in_left_m.resize( m_num_dims_m );
  m_strides_in_left_k.resize( m_num_dims_k );

  m_strides_in_right_c.resize( m_num_dims_c );
  m_strides_in_right_n.resize( m_num_dims_n );
  m_strides_in_right_k.resize( m_num_dims_k );

  m_strides_out_c.resize( m_num_dims_c );
  m_strides_out_m.resize( m_num_dims_m );
  m_strides_out_n.resize( m_num_dims_n );

  for( int64_t l_c = 0; l_c < m_num_dims_c; l_c++ ) {
    int64_t l_id = m_dim_ids_c[l_c];
    m_strides_in_left_c[l_c]  = l_map_id_stride_in_left.at(l_id);
    m_strides_in_right_c[l_c] = l_map_id_stride_in_right.at(l_id);
    m_strides_out_c[l_c]      = l_map_id_stride_out.at(l_id);
  }
  for( int64_t l_m = 0; l_m < m_num_dims_m; l_m++ ) {
    int64_t l_id = m_dim_ids_m[l_m];
    m_strides_in_left_m[l_m]  = l_map_id_stride_in_left.at(l_id);
    m_strides_out_m[l_m]      = l_map_id_stride_out.at(l_id);
  }
  for( int64_t l_n = 0; l_n < m_num_dims_n; l_n++ ) {
    int64_t l_id = m_dim_ids_n[l_n];
    m_strides_in_right_n[l_n] = l_map_id_stride_in_right.at(l_id);
    m_strides_out_n[l_n]      = l_map_id_stride_out.at(l_id);
  }
  for( int64_t l_k = 0; l_k < m_num_dims_k; l_k++ ) {
    int64_t l_id = m_dim_ids_k[l_k];
    m_strides_in_left_k[l_k]  = l_map_id_stride_in_left.at(l_id);
    m_strides_in_right_k[l_k]  = l_map_id_stride_in_right.at(l_id);
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
  else if( m_ktype_first_touch != UNDEFINED_KTYPE ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // inner kernel
  if( m_ktype_inner == MADD ) {
    if( l_dtype_all_fp32 ) {
      m_kernel_inner = &kernel_madd< float, float, float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel_inner = &kernel_madd< double, double, double >;
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
  else if( m_ktype_first_touch != UNDEFINED_KTYPE ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  m_compiled = true;

  return einsum_ir::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::BinaryContractionScalar::compile() {
  err_t l_err = compile( LEFT_NATIVE_RIGHT_NATIVE_OUT_NATIVE );
  return l_err;
}

void einsum_ir::backend::BinaryContractionScalar::contract( void const * i_tensor_in_left,
                                                            void const * i_tensor_in_right,
                                                            void       * io_tensor_out ) {

  ContractionLoopsSimple l_cont_loops;
  l_cont_loops.init( m_num_dims_c,
                     m_num_dims_m,
                     m_num_dims_n,
                     m_num_dims_k,
                     m_sizes_c.data(),
                     m_sizes_m.data(),
                     m_sizes_n.data(),
                     m_sizes_k.data(),
                     m_strides_in_left_c.data(),
                     m_strides_in_left_m.data(),
                     m_strides_in_left_k.data(),
                     m_strides_in_right_c.data(),
                     m_strides_in_right_n.data(),
                     m_strides_in_right_k.data(),
                     m_strides_out_c.data(),
                     m_strides_out_m.data(),
                     m_strides_out_n.data(),
                     ce_n_bytes( m_dtype_left ),
                     ce_n_bytes( m_dtype_right ),
                     ce_n_bytes( m_dtype_out ),
                     m_kernel_first_touch,
                     m_kernel_inner,
                     m_kernel_last_touch );

  l_cont_loops.contract( i_tensor_in_left,
                         i_tensor_in_right,
                         io_tensor_out );
}