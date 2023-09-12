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
          *m_dim_sizes_inner,
          *m_dim_sizes_outer_right,
          *m_dim_sizes_outer_left,
          *m_dim_sizes_outer_out,
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
    m_tensor_ordering = LEFT_BC_BM_BI_BK_IB_KB_MB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_OUT_NATIVE;

    // use standard GEMM kernels
    m_num_dims_cb = 0;

    // use consecutive M dimensions in output tensor as mb until target is reached
    m_num_dims_mb = 0;
    int64_t l_id_out = m_num_dims_out - 1;
    int64_t l_block_dim_size = 1;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == M &&
          l_block_dim_size < m_size_mb_gemm_target ) {
        int64_t l_dim_id              = m_dim_ids_out[l_id_out];
        int64_t l_dim_size_inner      = m_dim_sizes_inner->at(      l_dim_id );
        int64_t l_dim_size_outer_left = m_dim_sizes_outer_left->at( l_dim_id );
        int64_t l_dim_size_outer_out  = m_dim_sizes_outer_out->at(  l_dim_id );
        bool l_cont = l_dim_size_inner == l_dim_size_outer_left  &&
                      l_dim_size_inner == l_dim_size_outer_out;

        // only merge dim if M is stored contiguously
        if( m_num_dims_mb == 0 || l_cont ) {
          l_block_dim_size *= l_dim_size_inner;
          m_num_dims_mb++;
          l_id_out--;
        }
        else {
          break;
        }
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
        int64_t l_dim_id               = m_dim_ids_out[l_id_out];
        int64_t l_dim_size_inner       = m_dim_sizes_inner->at(       l_dim_id );
        int64_t l_dim_size_outer_right = m_dim_sizes_outer_right->at( l_dim_id );
        int64_t l_dim_size_outer_out   = m_dim_sizes_outer_out->at(   l_dim_id );
        int64_t l_cont = l_dim_size_inner == l_dim_size_outer_right &&
                         l_dim_size_inner == l_dim_size_outer_out;

        // only merge dim if N is store contiguously
        if( m_num_dims_nb == 0 || l_cont ) {
          l_block_dim_size *= l_dim_size_inner;
          m_num_dims_nb++;
          l_id_out--;
        }
        else {
          break;
        }
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
      int64_t l_dim_id               = m_dim_ids_k[l_id_k];
      int64_t l_dim_size_inner       = m_dim_sizes_inner->at(       l_dim_id );
      int64_t l_dim_size_outer_left  = m_dim_sizes_outer_left->at(  l_dim_id );
      int64_t l_dim_size_outer_right = m_dim_sizes_outer_right->at( l_dim_id );
      bool l_cont = l_dim_size_inner == l_dim_size_outer_left  &&
                    l_dim_size_inner == l_dim_size_outer_right;

      // only merge dim if K is stored contiguously
      if( m_num_dims_kb == 0 || l_cont ) {
        l_block_dim_size *= l_dim_size_inner;
        m_num_dims_kb++;
        l_id_k--;
      }
      else {
        break;
      }

      if( l_block_dim_size >= m_size_kb_gemm_target ) {
        break;
      }
    }
  }
  else if( m_dim_types_out[ m_num_dims_out - 1 ] == C ) {
    m_tensor_ordering = LEFT_BC_BM_BI_BK_IB_KB_MB_CB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_CB_OUT_NATIVE;

    // use consecutive C dimensions in output tensor as cb until target is reached
    m_num_dims_cb = 0;
    int64_t l_id_out = m_num_dims_out - 1;
    int64_t l_block_dim_size = 1;
    while( l_id_out >= 0 ) {
      if( m_dim_types_out[ l_id_out ] == C &&
          l_block_dim_size < m_size_cb_packed_gemm_target ) {
        int64_t l_dim_id               = m_dim_ids_out[l_id_out];
        int64_t l_dim_size_inner       = m_dim_sizes_inner->at(       l_dim_id );
        int64_t l_dim_size_outer_left  = m_dim_sizes_outer_left->at(  l_dim_id );
        int64_t l_dim_size_outer_right = m_dim_sizes_outer_right->at( l_dim_id );
        int64_t l_dim_size_outer_out   = m_dim_sizes_outer_out->at(   l_dim_id );
        bool l_cont = l_dim_size_inner == l_dim_size_outer_left  &&
                      l_dim_size_inner == l_dim_size_outer_right &&
                      l_dim_size_inner == l_dim_size_outer_out;

        // only merge dim if C is stored contiguously
        if( m_num_dims_cb == 0 || l_cont ) {
          l_block_dim_size *= l_dim_size_inner;
          m_num_dims_cb++;
          l_id_out--;
        }
        else {
          break;
        }
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
           *m_dim_sizes_outer_out,
           m_strides_left_c.data(),
           m_strides_left_m.data(),
           m_strides_left_k.data(),
           m_strides_left_i.data(),
           m_strides_right_c.data(),
           m_strides_right_n.data(),
           m_strides_right_k.data(),
           m_strides_right_j.data(),
           m_strides_out_c.data(),
           m_strides_out_m.data(),
           m_strides_out_n.data() );

  // number of K loop dims
  int64_t l_num_k_loop_dims = m_num_dims_k - m_num_dims_kb;

  // treat secondary I dimensions as K internally
  for( int64_t l_di_i = 0; l_di_i < m_num_dims_i; l_di_i++ ) {
    int64_t l_di_n = 0;
    l_err = link_secondary_to_primary( m_dim_ids_i[l_di_i],
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
    m_sizes_k.insert( m_sizes_k.begin() + l_num_k_loop_dims,
                      l_size_s );
    m_strides_left_k.insert( m_strides_left_k.begin() + l_num_k_loop_dims,
                             l_stride_s );
    m_strides_right_k.insert( m_strides_right_k.begin() + l_num_k_loop_dims,
                              l_stride_p );
    l_num_k_loop_dims++;
  }

  // treat secondary J dimensions as K internally
  for( int64_t l_di_j = 0; l_di_j < m_num_dims_j; l_di_j++ ) {
    int64_t l_di_m = 0;
    l_err = link_secondary_to_primary( m_dim_ids_j[l_di_j],
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
    m_sizes_k.insert( m_sizes_k.begin() + l_num_k_loop_dims,
                      l_size_s );
    m_strides_left_k.insert( m_strides_left_k.begin() + l_num_k_loop_dims,
                             l_stride_p );
    m_strides_right_k.insert( m_strides_right_k.begin() + l_num_k_loop_dims,
                              l_stride_s );
    l_num_k_loop_dims++;
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

  libxsmm_blasint l_lda = 1;
  libxsmm_blasint l_ldb = 1;
  libxsmm_blasint l_ldc = 1;

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

  // set leading dimensions
  // alternatives (l_m, l_k, l_m*l_r) have no purpose other than satisfying the jitter
  l_lda = m_num_dims_kb > 0 ? m_strides_left_k[  m_num_dims_k - 1 ] : l_m;
  l_ldb = m_num_dims_nb > 0 ? m_strides_right_n[ m_num_dims_n - 1 ] : l_k;
  l_ldc = m_num_dims_nb > 0 ? m_strides_out_n[   m_num_dims_n - 1 ] : l_m*l_r;

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

  if( m_tensor_ordering == LEFT_BC_BM_BI_BK_IB_KB_MB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_OUT_NATIVE ) {
    m_xmm_kernel_inner.gemm = libxsmm_dispatch_brgemm_v2( l_shape_brgemm,
                                                          l_flags_brgemm,
                                                          l_prefetch_flags_brgemm,
                                                          l_brconfig );
  }
  else if( m_tensor_ordering == LEFT_BC_BM_BI_BK_IB_KB_MB_CB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_CB_OUT_NATIVE ) {
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
                     m_strides_left_c.data(),
                     m_strides_left_m.data(),
                     m_strides_left_k.data(),
                     m_strides_right_c.data(),
                     m_strides_right_n.data(),
                     m_strides_right_k.data(),
                     m_strides_out_c.data(),
                     m_strides_out_m.data(),
                     m_strides_out_n.data(),
                     ce_n_bytes( m_dtype_left ),
                     ce_n_bytes( m_dtype_right ),
                     ce_n_bytes( m_dtype_out ),
                     m_xmm_kernel_first_touch,
                     m_xmm_kernel_inner.gemm,
                     m_xmm_kernel_last_touch );

  l_err = m_cont_loops.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return einsum_ir::COMPILATION_FAILED;
  }

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