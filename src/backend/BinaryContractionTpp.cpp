#include "BinaryContractionTpp.h"
#include "BinaryPrimitives.h"

#include "Unary.h"
#include "UnaryTpp.h"
#include <iostream>

#include <algorithm>

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
  err_t l_err = err_t::UNDEFINED_ERROR;

  // determine blocking type
  BinaryPrimitives l_bin_prim;
  l_bin_prim.init( m_dtype_comp,
                   backend_t::TPP );

  int64_t const * l_dim_ids_left_active  = m_dim_ids_permute_left  != nullptr ? m_dim_ids_permute_left  : m_dim_ids_left;
  int64_t const * l_dim_ids_right_active = m_dim_ids_permute_right != nullptr ? m_dim_ids_permute_right : m_dim_ids_right;

  // perform blocking
  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  l_err = l_bin_prim.blocking( primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                               m_num_dims_left,
                               m_num_dims_right,
                               m_num_dims_out,
                               l_dim_ids_left_active,
                               l_dim_ids_right_active,
                               m_dim_ids_out,
                               m_dim_sizes_inner,
                               nullptr,
                               nullptr,
                               nullptr,
                               &l_dim_ids_cb,
                               &l_dim_ids_mb,
                               &l_dim_ids_nb,
                               &l_dim_ids_kb );
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  // derive IDs of non-blocked dimensions
  std::vector< int64_t > l_dim_ids_bc;
  std::vector< int64_t > l_dim_ids_bm;
  std::vector< int64_t > l_dim_ids_bn;
  std::vector< int64_t > l_dim_ids_bk;

  for( int64_t l_di = 0; l_di < m_num_dims_c; l_di++ ) {
    if( std::find( l_dim_ids_cb.begin(), l_dim_ids_cb.end(), m_dim_ids_c[l_di] ) == l_dim_ids_cb.end() ) {
      l_dim_ids_bc.push_back( m_dim_ids_c[l_di] );
    }
  }
  for( int64_t l_di = 0; l_di < m_num_dims_m; l_di++ ) {
    if( std::find( l_dim_ids_mb.begin(), l_dim_ids_mb.end(), m_dim_ids_m[l_di] ) == l_dim_ids_mb.end() ) {
      l_dim_ids_bm.push_back( m_dim_ids_m[l_di] );
    }
  }
  for( int64_t l_di = 0; l_di < m_num_dims_n; l_di++ ) {
    if( std::find( l_dim_ids_nb.begin(), l_dim_ids_nb.end(), m_dim_ids_n[l_di] ) == l_dim_ids_nb.end() ) {
      l_dim_ids_bn.push_back( m_dim_ids_n[l_di] );
    }
  }
  for( int64_t l_di = 0; l_di < m_num_dims_k; l_di++ ) {
    if( std::find( l_dim_ids_kb.begin(), l_dim_ids_kb.end(), m_dim_ids_k[l_di] ) == l_dim_ids_kb.end() ) {
      l_dim_ids_bk.push_back( m_dim_ids_k[l_di] );
    }
  }

  // derive sizes of non-blocked dimensions
  m_sizes_bc.clear();
  m_sizes_bm.clear();
  m_sizes_bn.clear();
  m_sizes_bk.clear();

  for( std::size_t l_di = 0; l_di < l_dim_ids_bc.size(); l_di++ ) {
    int64_t l_dim_id = l_dim_ids_bc[l_di];
    m_sizes_bc.push_back( m_dim_sizes_inner->at( l_dim_id ) );
  }

  for( std::size_t l_di = 0; l_di < l_dim_ids_bm.size(); l_di++ ) {
    int64_t l_dim_id = l_dim_ids_bm[l_di];
    m_sizes_bm.push_back( m_dim_sizes_inner->at( l_dim_id ) );
  }

  for( std::size_t l_di = 0; l_di < l_dim_ids_bn.size(); l_di++ ) {
    int64_t l_dim_id = l_dim_ids_bn[l_di];
    m_sizes_bn.push_back( m_dim_sizes_inner->at( l_dim_id ) );
  }

  for( std::size_t l_di = 0; l_di < l_dim_ids_bk.size(); l_di++ ) {
    int64_t l_dim_id = l_dim_ids_bk[l_di];
    m_sizes_bk.push_back( m_dim_sizes_inner->at( l_dim_id ) );
  }

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
  for( std::size_t l_di = 0; l_di < l_dim_ids_bc.size(); l_di++ ) {
    int64_t l_dim_id = l_dim_ids_bc[l_di];
    m_strides_left_bc.push_back(    l_strides_left[    l_dim_id ] );
    m_strides_right_bc.push_back(   l_strides_right[   l_dim_id ] );
    m_strides_out_bc.push_back(     l_strides_out[     l_dim_id ] );
    m_strides_out_aux_bc.push_back( l_strides_out_aux[ l_dim_id ] );
  }

  for( std::size_t l_di = 0; l_di < l_dim_ids_bm.size(); l_di++ ) {
    int64_t l_dim_id = l_dim_ids_bm[l_di];
    m_strides_left_bm.push_back(    l_strides_left[    l_dim_id ] );
    m_strides_out_bm.push_back(     l_strides_out[     l_dim_id ] );
    m_strides_out_aux_bm.push_back( l_strides_out_aux[ l_dim_id ] );
  }

  for( std::size_t l_di = 0; l_di < l_dim_ids_bn.size(); l_di++ ) {
    int64_t l_dim_id = l_dim_ids_bn[l_di];
    m_strides_right_bn.push_back(   l_strides_right[   l_dim_id ] );
    m_strides_out_bn.push_back(     l_strides_out[     l_dim_id ] );
    m_strides_out_aux_bn.push_back( l_strides_out_aux[ l_dim_id ] );
  }

  for( std::size_t l_di = 0; l_di < l_dim_ids_bk.size(); l_di++ ) {
    int64_t l_dim_id = l_dim_ids_bk[l_di];
    m_strides_left_bk.push_back(  l_strides_left[  l_dim_id ] );
    m_strides_right_bk.push_back( l_strides_right[ l_dim_id ] );
  }

  int64_t l_size_packing_left = 0;
  int64_t l_size_packing_right = 0;
  UnaryTpp * l_unary_left = nullptr;
  UnaryTpp * l_unary_right = nullptr;

  if( m_dim_ids_permute_left != nullptr ){
    //determine packed dims (packed dims = kernel dims)
    std::vector< int64_t > l_dim_ids_packed;
    l_dim_ids_packed.reserve( l_dim_ids_cb.size() + l_dim_ids_mb.size() + l_dim_ids_kb.size() );
    l_dim_ids_packed.insert( l_dim_ids_packed.end(), l_dim_ids_kb.begin(), l_dim_ids_kb.end() );
    l_dim_ids_packed.insert( l_dim_ids_packed.end(), l_dim_ids_mb.begin(), l_dim_ids_mb.end() );
    l_dim_ids_packed.insert( l_dim_ids_packed.end(), l_dim_ids_cb.begin(), l_dim_ids_cb.end() );

    //determine input dims and input strides
    std::vector< int64_t > l_dim_ids_in;
    std::vector< int64_t > l_strides_in;
    l_dim_ids_in.reserve(l_dim_ids_packed.size());
    for( int64_t l_di = 0; l_di < m_num_dims_left; l_di++ ) {
      if( std::find( l_dim_ids_packed.begin(), l_dim_ids_packed.end(), m_dim_ids_left[l_di] ) != l_dim_ids_packed.end() ) {
        int64_t l_dim_id = m_dim_ids_left[l_di];
        l_dim_ids_in.push_back( l_dim_id );
        l_strides_in.push_back( l_strides_left[l_dim_id] );
      }
    }    
    //! unary operation
    l_unary_left = new UnaryTpp;
    l_unary_left->init( l_dim_ids_packed.size(),
                        m_dim_sizes_inner,
                        l_dim_ids_in.data(),
                        l_dim_ids_packed.data(),
                        l_strides_in.data(),
                        nullptr,
                        m_dtype_right,
                        m_dtype_right,
                        m_dtype_right,
                        kernel_t::COPY );
    
    l_err = l_unary_left->compile();
    if( l_err != einsum_ir::SUCCESS ) {
      return l_err;
    }
    l_unary_left->threading(1);

    //determine required memory
    l_size_packing_left = 1;
    for( int64_t l_di = 0; l_di < l_dim_ids_packed.size(); l_di++){
      l_size_packing_left *= m_dim_sizes_inner->at(l_dim_ids_packed[l_di]);
    }
    l_size_packing_left *= ce_n_bytes(m_dtype_left);
  }
  if( m_dim_ids_permute_right != nullptr ){
    //determine packed dims (packed dims = kernel dims)
    std::vector< int64_t > l_dim_ids_packed;
    l_dim_ids_packed.reserve( l_dim_ids_cb.size() + l_dim_ids_kb.size() + l_dim_ids_nb.size() );
    l_dim_ids_packed.insert( l_dim_ids_packed.end(), l_dim_ids_nb.begin(), l_dim_ids_nb.end() );
    l_dim_ids_packed.insert( l_dim_ids_packed.end(), l_dim_ids_kb.begin(), l_dim_ids_kb.end() );
    l_dim_ids_packed.insert( l_dim_ids_packed.end(), l_dim_ids_cb.begin(), l_dim_ids_cb.end() );
    
    //determine input dims and input strides
    std::vector< int64_t > l_dim_ids_in;
    std::vector< int64_t > l_strides_in;
    l_dim_ids_in.reserve(l_dim_ids_packed.size());
    for( int64_t l_di = 0; l_di < m_num_dims_right; l_di++ ) {
      if( std::find( l_dim_ids_packed.begin(), l_dim_ids_packed.end(), m_dim_ids_right[l_di] ) != l_dim_ids_packed.end() ) {
        int64_t l_dim_id = m_dim_ids_right[l_di];
        l_dim_ids_in.push_back( l_dim_id );
        l_strides_in.push_back( l_strides_right[l_dim_id] );
      }
    }

    //! unary operation
    l_unary_right = new UnaryTpp;
    l_unary_right->init( l_dim_ids_packed.size(),
                         m_dim_sizes_inner,
                         l_dim_ids_in.data(),
                         l_dim_ids_packed.data(),
                         l_strides_in.data(),
                         nullptr,
                         m_dtype_right,
                         m_dtype_right,
                         m_dtype_right,
                         kernel_t::COPY );
    
    l_err = l_unary_right->compile();
    if( l_err != einsum_ir::SUCCESS ) {
      return l_err;
    }
    l_unary_right->threading(1);

    //determine required memory
    l_size_packing_right = 1;
    for( int64_t l_di = 0; l_di < l_dim_ids_packed.size(); l_di++){
      l_size_packing_right *= m_dim_sizes_inner->at(l_dim_ids_packed[l_di]);
    }
    l_size_packing_right *= ce_n_bytes(m_dtype_right);
  }

  m_memory->reserve_thread_memory( l_size_packing_left + l_size_packing_right );

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

  for( std::size_t l_cb = 0; l_cb < l_dim_ids_cb.size(); l_cb++ ) {
    int64_t l_dim_id = l_dim_ids_cb[l_cb];
    l_r *= m_dim_sizes_inner->at( l_dim_id );
  }
  for( std::size_t l_mb = 0; l_mb < l_dim_ids_mb.size(); l_mb++ ) {
    int64_t l_dim_id = l_dim_ids_mb[l_mb];
    l_m *= m_dim_sizes_inner->at( l_dim_id );
  }
  for( std::size_t l_nb = 0; l_nb < l_dim_ids_nb.size(); l_nb++ ) {
    int64_t l_dim_id = l_dim_ids_nb[l_nb];
    l_n *= m_dim_sizes_inner->at( l_dim_id );
  }
  for( std::size_t l_kb = 0; l_kb < l_dim_ids_kb.size(); l_kb++ ) {
    int64_t l_dim_id = l_dim_ids_kb[l_kb];
    l_k *= m_dim_sizes_inner->at( l_dim_id );
  }

  // set leading dimensions
  l_lda = l_dim_ids_kb.size() > 0 && !l_size_packing_left  ? l_strides_left.at(  l_dim_ids_kb.back() ) : l_m*l_r;
  l_ldb = l_dim_ids_nb.size() > 0 && !l_size_packing_right ? l_strides_right.at( l_dim_ids_nb.back() ) : l_k*l_r;
  l_ldc = l_dim_ids_nb.size() > 0 ? l_strides_out.at(   l_dim_ids_nb.back() ) : l_m*l_r;

  // first-touch and last-touch shape
  libxsmm_meltw_unary_shape l_shape_single_touch = libxsmm_create_meltw_unary_shape( l_m*l_r,
                                                                                     l_n,
                                                                                     l_ldc,
                                                                                     l_ldc,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out,
                                                                                     l_xmm_dtype_out );

  // derive the leading dimension of the auxiliary output tensor
  int64_t l_ld_out_aux = l_dim_ids_nb.size() > 0 ? l_strides_out_aux.at( l_dim_ids_nb.back() ) : l_m*l_r;
  libxsmm_bitfield l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_bitfield l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_NONE;

  // column to matrix bcast
  if(         l_dim_ids_mb.size() > 0
           && l_dim_ids_nb.size() > 0
           && l_strides_out_aux[ l_dim_ids_mb.back() ] >  0
           && l_strides_out_aux[ l_dim_ids_nb.back() ] == 0 ) {
    l_ld_out_aux = l_m*l_r;
    l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
    l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
  }
  // row to matrix bcast
  else if(    l_dim_ids_mb.size() > 0
           && l_dim_ids_nb.size() > 0
           && l_strides_out_aux[ l_dim_ids_mb.back() ] == 0
           && l_strides_out_aux[ l_dim_ids_nb.back() ] >  0 ) {
    l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW;
    l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
  }
  // scalar to matrix bcast
  else if(    l_dim_ids_mb.size() > 0
           && l_dim_ids_nb.size() > 0
           && l_strides_out_aux[ l_dim_ids_mb.back() ] == 0
           && l_strides_out_aux[ l_dim_ids_nb.back() ] == 0 ) {
    l_ld_out_aux = l_m*l_r;
    l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
    l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
  }
  else if(    l_dim_ids_mb.size() == 0
           && l_dim_ids_nb.size() >  0
           && l_strides_out_aux[ l_dim_ids_nb.back() ] == 0 ) {
    l_flag_out_aux_unary  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
    l_flag_out_aux_binary = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
  }

  libxsmm_meltw_unary_shape l_shape_single_touch_aux_unary = libxsmm_create_meltw_unary_shape( l_m*l_r,
                                                                                               l_n,
                                                                                               l_ld_out_aux,
                                                                                               l_ldc,
                                                                                               l_xmm_dtype_out,
                                                                                               l_xmm_dtype_out,
                                                                                               l_xmm_dtype_out );

  libxsmm_meltw_binary_shape l_shape_single_touch_aux_binary = libxsmm_create_meltw_binary_shape( l_m*l_r,
                                                                                                  l_n,
                                                                                                  l_ldc,
                                                                                                  l_ld_out_aux,
                                                                                                  l_ldc,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out,
                                                                                                  l_xmm_dtype_out );

  // first touch kernel
  if( m_ktype_first_touch == ZERO ) {
    m_xmm_kernel_first_touch_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_XOR,
                                                                   l_shape_single_touch,
                                                                   LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }
  else if( m_ktype_first_touch == COPY ) {
    m_xmm_kernel_first_touch_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY,
                                                                   l_shape_single_touch_aux_unary,
                                                                   l_flag_out_aux_unary );
  }
  else if( m_ktype_first_touch == ADD ) {
    m_xmm_kernel_first_touch_binary = libxsmm_dispatch_meltw_binary( LIBXSMM_MELTW_TYPE_BINARY_ADD,
                                                                     l_shape_single_touch_aux_binary,
                                                                     l_flag_out_aux_binary );
  }
  else if( m_ktype_first_touch != UNDEFINED_KTYPE ) {
    return err_t::COMPILATION_FAILED;
  }

  // last touch kernel
  if( m_ktype_last_touch == RELU ) {
    m_xmm_kernel_last_touch_unary = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_RELU,
                                                                  l_shape_single_touch,
                                                                  LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }
  else if( m_ktype_last_touch == ADD ) {
    m_xmm_kernel_last_touch_binary = libxsmm_dispatch_meltw_binary( LIBXSMM_MELTW_TYPE_BINARY_ADD,
                                                                    l_shape_single_touch_aux_binary,
                                                                    l_flag_out_aux_binary );
  }
  else if( m_ktype_last_touch != UNDEFINED_KTYPE ) {
    return err_t::COMPILATION_FAILED;
  }

  // remove packed size form leading dimension (implicit in LIBXSMM)
  if( l_lda % l_r != 0 ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  if( l_ldb % l_r != 0 ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  if( l_ldc % l_r != 0 ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  l_lda /= l_r;
  l_ldb /= l_r;
  l_ldc /= l_r;

  // create main kernel
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

  if(    m_num_dims_out > 0
      && m_dim_types_out[ m_num_dims_out - 1 ] != dim_t::C ) {
    m_xmm_kernel_main.gemm = libxsmm_dispatch_brgemm( l_shape_brgemm,
                                                      l_flags_brgemm,
                                                      l_prefetch_flags_brgemm,
                                                      l_brconfig );
  }
  else {
    m_xmm_kernel_main.gemm = libxsmm_create_packed_gemm( l_shape_brgemm,
                                                         l_flags_brgemm,
                                                         l_prefetch_flags_brgemm,
                                                         l_r );
  }

  // check for existing kernels
  if( m_xmm_kernel_first_touch_unary == nullptr ) {
    if(    m_ktype_first_touch != kernel_t::UNDEFINED_KTYPE
        && m_ktype_first_touch != kernel_t::ADD ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  if( m_xmm_kernel_first_touch_binary == nullptr ) {
    if(    m_ktype_first_touch != kernel_t::UNDEFINED_KTYPE
        && m_ktype_first_touch != kernel_t::ZERO
        && m_ktype_first_touch != kernel_t::COPY
        && m_ktype_first_touch != kernel_t::RELU ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  if(    m_ktype_main             != UNDEFINED_KTYPE
      && m_xmm_kernel_main.gemm   == nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }
  if( m_xmm_kernel_last_touch_unary == nullptr ) {
    if(    m_ktype_last_touch != kernel_t::UNDEFINED_KTYPE
        && m_ktype_last_touch != kernel_t::ADD ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  if( m_xmm_kernel_last_touch_binary == nullptr ) {
    if(    m_ktype_last_touch != kernel_t::UNDEFINED_KTYPE
        && m_ktype_last_touch != kernel_t::RELU ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }

  // contraction loop interface
  m_cont_loops.init( l_dim_ids_bc.size(),
                     l_dim_ids_bm.size(),
                     l_dim_ids_bn.size(),
                     l_dim_ids_bk.size(),
                     m_sizes_bc.data(),
                     m_sizes_bm.data(),
                     m_sizes_bn.data(),
                     m_sizes_bk.data(),
                     m_strides_left_bc.data(),
                     m_strides_left_bm.data(),
                     m_strides_left_bk.data(),
                     m_strides_right_bc.data(),
                     m_strides_right_bn.data(),
                     m_strides_right_bk.data(),
                     m_strides_out_aux_bc.data(),
                     m_strides_out_aux_bm.data(),
                     m_strides_out_aux_bn.data(),
                     m_strides_out_bc.data(),
                     m_strides_out_bm.data(),
                     m_strides_out_bn.data(),
                     ce_n_bytes( m_dtype_left ),
                     ce_n_bytes( m_dtype_right ),
                     ce_n_bytes( m_dtype_out ),
                     m_ktype_first_touch,
                     m_ktype_main,
                     m_ktype_last_touch,
                     m_xmm_kernel_first_touch_unary,
                     m_xmm_kernel_first_touch_binary,
                     m_xmm_kernel_main.gemm,
                     m_xmm_kernel_last_touch_unary,
                     m_xmm_kernel_last_touch_binary,
                     l_unary_left,
                     l_unary_right,
                     m_memory,
                     l_size_packing_left,
                     l_size_packing_right );

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

void einsum_ir::backend::BinaryContractionTpp::contract( void const * i_tensor_left,
                                                         void const * i_tensor_right,
                                                         void const * i_tensor_out_aux,
                                                         void       * io_tensor_out ) {
  m_cont_loops.contract( i_tensor_left,
                         i_tensor_right,
                         i_tensor_out_aux,
                         io_tensor_out );
}

void einsum_ir::backend::BinaryContractionTpp::contract( void const * i_tensor_left,
                                                         void const * i_tensor_right,
                                                         void       * io_tensor_out ) {
  contract( i_tensor_left,
            i_tensor_right,
            nullptr,
            io_tensor_out );
}