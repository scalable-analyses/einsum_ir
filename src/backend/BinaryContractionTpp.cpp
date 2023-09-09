#include "BinaryContractionTpp.h"

libxsmm_datatype einsum_ir::backend::BinaryContractionTpp::dtype_to_libxsmm( data_t i_dtype ) {
  if( i_dtype == FP32 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F32;
  }
  else if( i_dtype == FP64 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F64;
  }

  return libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED;
}

einsum_ir::err_t einsum_ir::backend::BinaryContractionTpp::compile() {
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
          *m_dim_sizes,
          m_dim_ids_right_native,
          m_dim_ids_left_native,
          m_dim_ids_out,
          m_dtype_right,
          m_dtype_left,
          m_dtype_comp,
          m_dtype_out,
          m_ktype_first_touch,
          m_ktype_inner,
          m_ktype_last_touch );

    BinaryContraction::compile_base();

    m_tensors_in_swapped = true;
  }

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

  // derive parameters for reordering of input dimensions
  if( m_dim_types_out[ m_num_dims_out - 1 ] == M ) {
    m_tensor_ordering = LEFT_BC_BM_BK_KB_MB_RIGHT_BC_BN_BK_NB_KB_OUT_NATIVE;

    // use standard GEMM kernels
    m_num_dims_cb = 0;

    // use consecutive M dimensions in output tensor as mb until target is reached
    m_num_dims_mb = 0;
    int64_t l_id_out = m_num_dims_out - 1;
    int64_t l_block_dim_size = 1;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == M &&
          l_block_dim_size < m_size_mb_gemm_target ) {
        int64_t l_dim_id = m_dim_ids_out[l_id_out];
        l_block_dim_size *= m_dim_sizes->at( l_dim_id );

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

    // use consecutive N dimensions in output tensor as nb until target is reached
    m_num_dims_nb = 0;
    l_block_dim_size = 1;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == N &&
          l_block_dim_size < m_size_nb_gemm_target ) {
        int64_t l_dim_id = m_dim_ids_out[l_id_out];
        l_block_dim_size *= m_dim_sizes->at( l_dim_id );

        m_num_dims_nb++;
        l_id_out--;
      }
      else {
        break;
      }
    }

    // determine number of K dimensions which reach the target (kb)
    int64_t l_id_k = m_num_dims_k - 1;
    m_num_dims_kb = 0;
    l_block_dim_size = 1;
    while( l_id_k >= 0 ) {
      l_block_dim_size *= m_dim_sizes->at( m_dim_ids_k[l_id_k] );

      m_num_dims_kb++;
      l_id_k--;

      if( l_block_dim_size >= m_size_kb_gemm_target ) {
        break;
      }
    }
  }
  else if( m_dim_types_out[ m_num_dims_out - 1 ] == C ) {
    m_tensor_ordering = LEFT_BC_BM_BK_KB_MB_CB_RIGHT_BC_BN_BK_NB_KB_CB_OUT_NATIVE;

    // use consecutive C dimensions in output tensor as cb until target is reached
    m_num_dims_cb = 0;
    int64_t l_id_out = m_num_dims_out - 1;
    int64_t l_block_dim_size = 1;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == C &&
          l_block_dim_size < m_size_cb_packed_gemm_target ) {
        int64_t l_dim_id = m_dim_ids_out[l_id_out];
        l_block_dim_size *= m_dim_sizes->at( l_dim_id );

        m_num_dims_cb++;
        l_id_out--;
      }
      else {
        break;
      }
    }

    // remaining dimension types
    m_num_dims_mb = (m_num_dims_m > 0) ? 1 : 0;
    m_num_dims_nb = (m_num_dims_n > 0) ? 1 : 0;
    m_num_dims_kb = (m_num_dims_k > 0) ? 1 : 0;
  }
  else {
    return einsum_ir::COMPILATION_FAILED;
  }

  // reorder input dimensions
  err_t l_err = order_dims_in( m_tensor_ordering,
                               m_num_dims_c,
                               m_num_dims_m,
                               m_num_dims_n,
                               m_num_dims_k,
                               m_num_dims_cb,
                               m_num_dims_mb,
                               m_num_dims_nb,
                               m_num_dims_kb,
                               m_dim_ids_c.data(),
                               m_dim_ids_m.data(),
                               m_dim_ids_n.data(),
                               m_dim_ids_k.data(),
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
  std::map< int64_t, int64_t > l_map_id_stride_in_left;
  std::map< int64_t, int64_t > l_map_id_stride_in_right;
  std::map< int64_t, int64_t > l_map_id_stride_out;

  strides( m_num_dims_left,
           l_dim_ids_left,
           *m_dim_sizes,
           l_map_id_stride_in_left );

  strides( m_num_dims_right,
           l_dim_ids_right,
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

  // libxsmm parameters
  libxsmm_blasint l_m = 1;
  libxsmm_blasint l_n = 1;
  libxsmm_blasint l_k = 1;
  libxsmm_blasint l_r = 1;

  libxsmm_blasint l_lda = 0;
  libxsmm_blasint l_ldb = 0;
  libxsmm_blasint l_ldc = 0;

  for( int64_t l_cb = 0; l_cb < m_num_dims_cb; l_cb++ ) {
    l_r *= m_sizes_c[ m_num_dims_c - 1 - l_cb ];
  }
  for( int64_t l_mb = 0; l_mb < m_num_dims_mb; l_mb++ ) {
    l_m *= m_sizes_m[ m_num_dims_m - 1 - l_mb ];
  }
  for( int64_t l_nb = 0; l_nb < m_num_dims_nb; l_nb++ ) {
    l_n *= m_sizes_n[ m_num_dims_n - 1 - l_nb ];
  }
  for( int64_t l_kb = 0; l_kb < m_num_dims_kb; l_kb++ ) {
    l_k *= m_sizes_k[ m_num_dims_k - 1 - l_kb ];
  }

  l_lda = l_m;
  l_ldb = l_k;
  l_ldc = m_num_dims_nb > 0 ? m_strides_out_n[ m_num_dims_n - 1 ] : l_m*l_r;

  // first-touch and last-touch shape
  libxsmm_meltw_unary_shape l_shape_single_touch = libxsmm_create_meltw_unary_shape( l_m*l_r,
                                                                                     l_n,
                                                                                     l_ldc,
                                                                                     l_ldc,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out );

  // first touch kernel
  if( m_ktype_first_touch == ZERO ) {
    m_xmm_kernel_first_touch = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR,
                                                                l_shape_single_touch,
                                                                LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }
  else if( m_ktype_first_touch != UNDEFINED_KTYPE ) {
    return err_t::COMPILATION_FAILED;
  }

  // last touch kernel
  if( m_ktype_last_touch == RELU ) {
    m_xmm_kernel_last_touch = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU,
                                                               l_shape_single_touch,
                                                               LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }
  else if( m_ktype_last_touch != UNDEFINED_KTYPE ) {
    return err_t::COMPILATION_FAILED;
  }

  // remove blocked C dimension for inner kernel from leading output dimensions which is implicit in LIBXSMM
  if( l_ldc % l_r != 0 ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  l_ldc /= l_r;

  // create inner kernel
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
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
  l_brconfig.br_stride_a_hint = 0;
  l_brconfig.br_stride_b_hint = 0;
  l_brconfig.br_unroll_hint = 0;

  if( m_tensor_ordering == LEFT_BC_BM_BK_KB_MB_RIGHT_BC_BN_BK_NB_KB_OUT_NATIVE ) {
    m_xmm_kernel_inner.gemm = libxsmm_dispatch_brgemm_v2( l_shape_brgemm,
                                                          l_flags_brgemm,
                                                          l_prefetch_flags_brgemm,
                                                          l_brconfig );
  }
  else if( m_tensor_ordering == LEFT_BC_BM_BK_KB_MB_CB_RIGHT_BC_BN_BK_NB_KB_CB_OUT_NATIVE ) {
    m_xmm_kernel_inner.gemm = libxsmm_create_packed_gemm( l_shape_brgemm,
                                                          l_flags_brgemm,
                                                          l_prefetch_flags_brgemm,
                                                          l_r );
  }
  else {
    return err_t::COMPILATION_FAILED;
  }

  // check for existing kernels
  if(    m_ktype_first_touch      != UNDEFINED_KTYPE
      && m_xmm_kernel_first_touch == nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  if(    m_ktype_inner            != UNDEFINED_KTYPE
      && m_xmm_kernel_inner.gemm == nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  if(    m_ktype_last_touch       != UNDEFINED_KTYPE
      && m_xmm_kernel_last_touch  == nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // contraction loop interface
  m_cont_loops.init( m_num_dims_c-m_num_dims_cb,
                     m_num_dims_m-m_num_dims_mb,
                     m_num_dims_n-m_num_dims_nb,
                     m_num_dims_k-m_num_dims_kb,
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
                     m_xmm_kernel_first_touch,
                     m_xmm_kernel_inner.gemm,
                     m_xmm_kernel_last_touch );

  m_compiled = true;

  return einsum_ir::SUCCESS;
}

void einsum_ir::backend::BinaryContractionTpp::threading( int64_t i_num_tasks_target  ) {
  m_cont_loops.threading( i_num_tasks_target );
}

void einsum_ir::backend::BinaryContractionTpp::contract( void const * i_tensor_in_left,
                                                         void const * i_tensor_in_right,
                                                         void       * io_tensor_out ) {
  void const * l_tensor_left = i_tensor_in_left;
  void const * l_tensor_right = i_tensor_in_right;
  if( m_tensors_in_swapped ) {
    l_tensor_left = i_tensor_in_right;
    l_tensor_right = i_tensor_in_left;
  }

  m_cont_loops.contract( l_tensor_left,
                         l_tensor_right,
                         io_tensor_out );
}