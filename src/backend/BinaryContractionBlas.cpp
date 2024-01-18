#include "BinaryContractionBlas.h"

einsum_ir::err_t einsum_ir::backend::BinaryContractionBlas::compile() {
  BinaryContraction::compile_base();

  // check if inputs have to be swapped
  bool l_swap_inputs = false;

  // stride-1 output dimension is N: swap required
  if( m_dim_types_out[ m_num_dims_out - 1 ] == N  ) {
    l_swap_inputs = true;
  }
  // stride-1 output dimension is C: swap depends on first non-C dimension
  else if( m_dim_types_out[ m_num_dims_out - 1 ] == C ) {
    int64_t l_id_out = m_num_dims_out - 1;

    while( l_id_out >= 0 ) {
      // no swap if first non-C dim is M
      if( m_dim_types_out[ l_id_out ] == M ) {
        break;
      }
      // swap if first non-C is N
      else if( m_dim_types_out[ l_id_out ] == N ) {
        l_swap_inputs = true;
        break;
      }
      l_id_out--;
    }
  }

  // swap left and right tensors if required
  if( l_swap_inputs ) {
    init( m_num_dims_right,
          m_num_dims_left,
          m_num_dims_out,
          m_dim_sizes_inner,
          m_dim_sizes_outer_right,
          m_dim_sizes_outer_left,
          m_dim_sizes_outer_out_aux,
          m_dim_sizes_outer_out,
          m_stride_multipliers_right,
          m_stride_multipliers_left,
          m_stride_multipliers_out,
          m_dim_ids_right_native,
          m_dim_ids_left_native,
          m_dim_ids_out,
          m_dim_link_s_to_p,
          m_dtype_right,
          m_dtype_left,
          m_dtype_comp,
          m_dtype_out,
          m_ktype_first_touch,
          m_ktype_main,
          m_ktype_last_touch );

    BinaryContraction::compile_base();

    m_tensors_in_swapped = true;
  }

  // abort if auxiliary output tensor is used
  if( m_dim_sizes_outer_out_aux != nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // check that the outer dimensions match the inner ones
  for( int64_t l_di = 0; l_di < m_num_dims_left; l_di++ ) {
    int64_t l_dim_id         = m_dim_ids_left_native[l_di];
    int64_t l_dim_size_inner = m_dim_sizes_inner->at(      l_dim_id );
    int64_t l_dim_size_outer = m_dim_sizes_outer_left->at( l_dim_id );

    if( l_dim_size_inner != l_dim_size_outer ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  for( int64_t l_di = 0; l_di < m_num_dims_right; l_di++ ) {
    int64_t l_dim_id         = m_dim_ids_right_native[l_di];
    int64_t l_dim_size_inner = m_dim_sizes_inner->at(      l_dim_id );
    int64_t l_dim_size_outer = m_dim_sizes_outer_right->at( l_dim_id );

    if( l_dim_size_inner != l_dim_size_outer ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  for( int64_t l_di = 0; l_di < m_num_dims_out; l_di++ ) {
    int64_t l_dim_id          = m_dim_ids_out[l_di];
    int64_t l_dim_size_inner  = m_dim_sizes_inner->at( l_dim_id );
    int64_t l_dim_size_outer  = m_dim_sizes_outer_out->at( l_dim_id );

    if( l_dim_size_inner != l_dim_size_outer ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }

  // derive parameters for reordering of input dimensions
  if( m_dim_types_out[ m_num_dims_out - 1 ] == dim_t::M ) {
    m_tensor_ordering = LEFT_BC_BM_BI_BK_IB_KB_MB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_OUT_NATIVE;

    // use standard GEMM kernels
    m_num_dims_cb = 0;
    int64_t l_id_out = 0;

    // merge consecutive M dimensions
    m_num_dims_mb = 0;
    l_id_out = m_num_dims_out - 1;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == dim_t::M ) {
        m_num_dims_mb++;
        l_id_out--;
      }
      else {
        break;
      }
    }

    // jump until we reach an N dimension
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] != dim_t::N ) {
        l_id_out--;
      }
      else {
        break;
      }
    }

    // merge consecutive N dimensions
    m_num_dims_nb = 0;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == dim_t::N ) {
        m_num_dims_nb++;
        l_id_out--;
      }
      else {
        break;
      }
    }

    // merge consecutive K dimensions
    int64_t l_id_k = m_num_dims_k - 1;
    m_num_dims_kb = 0;
    while( l_id_k >= 0 ) {
      m_num_dims_kb++;
      l_id_k--;
    }
  }
  else if( m_dim_types_out[ m_num_dims_out - 1 ] == dim_t::C ) {
    m_tensor_ordering = tenord_t::LEFT_BC_BM_BI_BK_CB_KB_MB_RIGHT_BC_BN_BJ_BK_CB_NB_KB_OUT_NATIVE;

    int64_t l_id_out = m_num_dims_out - 1;

    // merge consecutive C dimensions
    m_num_dims_cb = 0;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == dim_t::C ) {
        m_num_dims_cb++;
        l_id_out--;
      }
      else {
        break;
      }
    }

    // merge consecutive M dimensions
    m_num_dims_mb = 0;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == dim_t::M ) {
        m_num_dims_mb++;
        l_id_out--;
      }
      else {
        break;
      }
    }

    // jump until we reach an N dimension
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] != N ) {
        l_id_out--;
      }
      else {
        break;
      }
    }

    // merge consecutive N dimensions
    m_num_dims_nb = 0;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == dim_t::N ) {
        m_num_dims_nb++;
        l_id_out--;
      }
      else {
        break;
      }
    }

    // merge K dimensions
    int64_t l_id_k = m_num_dims_k - 1;
    m_num_dims_kb = 0;
    while( l_id_k >= 0 ) {
      m_num_dims_kb++;
      l_id_k--;
    }
  }
  else {
    return einsum_ir::COMPILATION_FAILED;
  }

  // reorder input dimensions
  m_dim_ids_left_ordered.resize( m_num_dims_left );
  m_dim_ids_left_ordered.insert( m_dim_ids_left_ordered.begin(),
                                 m_dim_ids_left_native,
                                 m_dim_ids_left_native + m_num_dims_left );

  m_dim_ids_right_ordered.resize( m_num_dims_right );
  m_dim_ids_right_ordered.insert( m_dim_ids_right_ordered.begin(),
                                  m_dim_ids_right_native,
                                  m_dim_ids_right_native + m_num_dims_right );

  int64_t * l_dim_ids_left  = m_dim_ids_left_ordered.data();
  int64_t * l_dim_ids_right = m_dim_ids_right_ordered.data();

  err_t l_err = order_dims_in( m_tensor_ordering,
                               m_num_dims_c,
                               m_num_dims_m,
                               m_num_dims_n,
                               m_num_dims_k,
                               m_num_dims_i,
                               m_num_dims_j,
                               m_num_dims_cb,
                               m_num_dims_mb,
                               m_num_dims_nb,
                               m_num_dims_kb,
                               0,
                               0,
                               m_dim_ids_c.data(),
                               m_dim_ids_m.data(),
                               m_dim_ids_n.data(),
                               m_dim_ids_k.data(),
                               m_dim_ids_i.data(),
                               m_dim_ids_j.data(),
                               l_dim_ids_left,
                               l_dim_ids_right );
  if( l_err != SUCCESS ) {
    return l_err;
  }

  // check if the reordering changed the data layout of the input tensors
  for( int64_t l_le = 0; l_le < m_num_dims_left; l_le++ ) {
    if( m_dim_ids_left_native[l_le] != l_dim_ids_left[l_le] ) {
      m_tensors_in_reordered = true;
      break;
    }
  }
  for( int64_t l_ri = 0; l_ri < m_num_dims_right; l_ri++ ) {
    if( m_dim_ids_right_native[l_ri] != l_dim_ids_right[l_ri] ) {
      m_tensors_in_reordered = true;
      break;
    }
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
           m_dim_sizes_outer_left,
           m_dim_sizes_outer_right,
           m_dim_sizes_outer_out_aux,
           m_dim_sizes_outer_out,
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

  // check that no I or J dimensions are present
  if( m_num_dims_i > 0 || m_num_dims_j > 0 ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // check that stride multiplies are not used
  if(    m_stride_multipliers_left  != nullptr
      || m_stride_multipliers_right != nullptr
      || m_stride_multipliers_out   != nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // derive BLAS parameters
  int64_t l_blas_size_c = 1;
  int64_t l_blas_size_m = 1;
  int64_t l_blas_size_n = 1;
  int64_t l_blas_size_k = 1;

  for( int64_t l_cb = 0; l_cb < m_num_dims_cb; l_cb++ ) {
    l_blas_size_c *= m_sizes_c[ m_num_dims_c - 1 - l_cb ];
  }
  for( int64_t l_mb = 0; l_mb < m_num_dims_mb; l_mb++ ) {
    l_blas_size_m *= m_sizes_m[ m_num_dims_m - 1 - l_mb ];
  }
  for( int64_t l_nb = 0; l_nb < m_num_dims_nb; l_nb++ ) {
    l_blas_size_n *= m_sizes_n[ m_num_dims_n - 1 - l_nb ];
  }
  for( int64_t l_kb = 0; l_kb < m_num_dims_kb; l_kb++ ) {
    l_blas_size_k *= m_sizes_k[ m_num_dims_k - 1 - l_kb ];
  }

  // set leading dimensions
  // alternatives (l_m, l_k, l_m*l_r) have no purpose other than satisfying the BLAS implementations
  int64_t l_blas_ld_a = m_num_dims_kb > 0 ? m_strides_left_k[  m_num_dims_k - 1 ] : l_blas_size_m;
  int64_t l_blas_ld_b = m_num_dims_nb > 0 ? m_strides_right_n[ m_num_dims_n - 1 ] : l_blas_size_k;
  int64_t l_blas_ld_c = m_num_dims_nb > 0 ? m_strides_out_n[   m_num_dims_n - 1 ] : l_blas_size_m*l_blas_size_c;

  // check that the same data type is used everywhere
  if(    m_dtype_comp != m_dtype_left
      || m_dtype_comp != m_dtype_right
      || m_dtype_comp != m_dtype_out ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // check supported kernel types
  if(    m_ktype_first_touch != einsum_ir::kernel_t::UNDEFINED_KTYPE
      && m_ktype_first_touch != einsum_ir::kernel_t::ZERO ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  if( m_ktype_main != einsum_ir::kernel_t::MADD ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  if( m_ktype_last_touch != einsum_ir::kernel_t::UNDEFINED_KTYPE ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // init contraction loops
  m_cont_loops.init( m_num_dims_c-m_num_dims_cb,
                     m_num_dims_m-m_num_dims_mb,
                     m_num_dims_n-m_num_dims_nb,
                     m_num_dims_k-m_num_dims_kb,
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
                     m_dtype_comp,
                     false,
                     false,
                     false,
                     l_blas_size_c,
                     l_blas_size_m,
                     l_blas_size_n,
                     l_blas_size_k,
                     l_blas_ld_a,
                     l_blas_ld_b,
                     l_blas_ld_c,
                     1.0,
                     1.0,
                     m_ktype_first_touch,
                     m_ktype_last_touch );

  l_err = m_cont_loops.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  m_compiled = true;

  return einsum_ir::SUCCESS;
}

void einsum_ir::backend::BinaryContractionBlas::threading( int64_t i_num_tasks_target  ) {
  m_cont_loops.threading( i_num_tasks_target );
}

void einsum_ir::backend::BinaryContractionBlas::contract( void const * i_tensor_left,
                                                          void const * i_tensor_right,
                                                          void const * i_tensor_out_aux,
                                                          void       * io_tensor_out ) {
  void const * l_tensor_left = i_tensor_left;
  void const * l_tensor_right = i_tensor_right;
  if( m_tensors_in_swapped ) {
    l_tensor_left = i_tensor_right;
    l_tensor_right = i_tensor_left;
  }

  m_cont_loops.contract( l_tensor_left,
                         l_tensor_right,
                         i_tensor_out_aux,
                         io_tensor_out );
}

void einsum_ir::backend::BinaryContractionBlas::contract( void const * i_tensor_left,
                                                          void const * i_tensor_right,
                                                          void       * io_tensor_out ) {
  contract( i_tensor_left,
            i_tensor_right,
            nullptr,
            io_tensor_out );
}