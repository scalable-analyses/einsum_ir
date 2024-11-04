#include "BinaryContractionTpp.h"
#include "BinaryPrimitives.h"
#include "ContractionPackingTpp.h"

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

  std::map < int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( m_dim_sizes_inner->begin(), m_dim_sizes_inner->end());
  //determine loop execution order
  if( m_loop_ids_ext == nullptr ){
    l_bin_prim.compileLoopOrder(m_dim_types,
                                l_dim_sizes,
                                l_strides_left,
                                l_strides_right,
                                l_strides_out,
                                m_dim_ids_c,
                                m_dim_ids_m,
                                m_dim_ids_n,
                                m_dim_ids_k,
                                l_dim_ids_cb,
                                l_dim_ids_mb,
                                l_dim_ids_nb,
                                l_dim_ids_kb,
                                m_loop_ids_int);
  }
  else{
    std::vector< int64_t > l_dim_ids_kernel;
    l_dim_ids_kernel.reserve( l_dim_ids_cb.size() + l_dim_ids_mb.size() + l_dim_ids_nb.size() + l_dim_ids_kb.size() );
    l_dim_ids_kernel.insert( l_dim_ids_kernel.end(), l_dim_ids_cb.begin(), l_dim_ids_cb.end() );
    l_dim_ids_kernel.insert( l_dim_ids_kernel.end(), l_dim_ids_nb.begin(), l_dim_ids_nb.end() );
    l_dim_ids_kernel.insert( l_dim_ids_kernel.end(), l_dim_ids_mb.begin(), l_dim_ids_mb.end() );
    l_dim_ids_kernel.insert( l_dim_ids_kernel.end(), l_dim_ids_kb.begin(), l_dim_ids_kb.end() );
    for( std::size_t l_di = 0; l_di < l_dim_ids_kernel.size(); l_di++){
      auto l_found = std::find( m_loop_ids_int.begin(), m_loop_ids_int.end(), l_dim_ids_kernel[l_di] );
      if( l_found != m_loop_ids_int.end() ) {
        m_loop_ids_int.erase(l_found);
      }
    }
  }

  std::vector< int64_t > l_dim_ids_packed_left;
  std::vector< int64_t > l_dim_ids_packed_right;
  ContractionPackingTpp * l_packing = nullptr;
  if( m_dim_ids_permute_left != nullptr ){
    l_dim_ids_packed_left.reserve( l_dim_ids_cb.size() + l_dim_ids_mb.size() + l_dim_ids_kb.size() );
    l_dim_ids_packed_left.insert( l_dim_ids_packed_left.end(), l_dim_ids_kb.begin(), l_dim_ids_kb.end() );
    l_dim_ids_packed_left.insert( l_dim_ids_packed_left.end(), l_dim_ids_mb.begin(), l_dim_ids_mb.end() );
    l_dim_ids_packed_left.insert( l_dim_ids_packed_left.end(), l_dim_ids_cb.begin(), l_dim_ids_cb.end() );
  }
  if( m_dim_ids_permute_right != nullptr ){
    l_dim_ids_packed_right.reserve( l_dim_ids_cb.size() + l_dim_ids_kb.size() + l_dim_ids_nb.size() );
    l_dim_ids_packed_right.insert( l_dim_ids_packed_right.end(), l_dim_ids_nb.begin(), l_dim_ids_nb.end() );
    l_dim_ids_packed_right.insert( l_dim_ids_packed_right.end(), l_dim_ids_kb.begin(), l_dim_ids_kb.end() );
    l_dim_ids_packed_right.insert( l_dim_ids_packed_right.end(), l_dim_ids_cb.begin(), l_dim_ids_cb.end() );
  }


  if( m_dim_ids_permute_left != nullptr || m_dim_ids_permute_right != nullptr ) {
    //compile packing
    l_packing = new ContractionPackingTpp;
    l_packing->init( m_num_dims_left,
                    m_num_dims_right,
                    &l_dim_sizes,
                    &l_strides_left,
                    &l_strides_right,
                    &m_dim_types,
                    m_dim_ids_left,
                    m_dim_ids_right,
                    &l_dim_ids_packed_left,
                    &l_dim_ids_packed_right,
                    &m_loop_ids_int,
                    m_dtype_left,
                    m_dtype_out,
                    m_memory );
    l_packing->compile();

    //update strides
    for( auto const& elem : l_packing->m_strides_packed_left ) {
      l_strides_left.insert_or_assign(elem.first, elem.second);
    }
    for( auto const& elem : l_packing->m_strides_packed_right ) {
      l_strides_right.insert_or_assign(elem.first, elem.second);
    }
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

  for( std::size_t l_cb = 0; l_cb < l_dim_ids_cb.size(); l_cb++ ) {
    int64_t l_dim_id = l_dim_ids_cb[l_cb];
    l_r *= l_dim_sizes.at( l_dim_id );
  }
  for( std::size_t l_mb = 0; l_mb < l_dim_ids_mb.size(); l_mb++ ) {
    int64_t l_dim_id = l_dim_ids_mb[l_mb];
    l_m *= l_dim_sizes.at( l_dim_id );
  }
  for( std::size_t l_nb = 0; l_nb < l_dim_ids_nb.size(); l_nb++ ) {
    int64_t l_dim_id = l_dim_ids_nb[l_nb];
    l_n *= l_dim_sizes.at( l_dim_id );
  }
  for( std::size_t l_kb = 0; l_kb < l_dim_ids_kb.size(); l_kb++ ) {
    int64_t l_dim_id = l_dim_ids_kb[l_kb];
    l_k *= l_dim_sizes.at( l_dim_id );
  }

  // set leading dimensions
  l_lda = l_dim_ids_kb.size() > 0 ? l_strides_left.at(  l_dim_ids_kb.back() ) : l_m*l_r;
  l_ldb = l_dim_ids_nb.size() > 0 ? l_strides_right.at( l_dim_ids_nb.back() ) : l_k*l_r;
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
  m_cont_loops.init( &l_dim_sizes,
                     &l_strides_left,
                     &l_strides_right,
                     &l_strides_out_aux,
                     &l_strides_out,
                     &m_dim_types,
                     &m_loop_ids_int,
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
                     l_packing );

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