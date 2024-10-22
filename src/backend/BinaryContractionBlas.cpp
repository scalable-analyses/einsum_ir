#include "BinaryContractionBlas.h"
#include "BinaryPrimitives.h"

#include <algorithm>

einsum_ir::err_t einsum_ir::backend::BinaryContractionBlas::compile() {
  err_t l_err = err_t::UNDEFINED_ERROR;

  l_err = BinaryContraction::compile_base();
  if( l_err != einsum_ir::SUCCESS ) {
    return l_err;
  }

  // abort if auxiliary output tensor is used
  if( m_dim_sizes_outer_out_aux != nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // check that the outer dimensions match the inner ones
  for( int64_t l_di = 0; l_di < m_num_dims_left; l_di++ ) {
    int64_t l_dim_id         = m_dim_ids_left[l_di];
    int64_t l_dim_size_inner = m_dim_sizes_inner->at(      l_dim_id );
    int64_t l_dim_size_outer = m_dim_sizes_outer_left->at( l_dim_id );

    if( l_dim_size_inner != l_dim_size_outer ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  for( int64_t l_di = 0; l_di < m_num_dims_right; l_di++ ) {
    int64_t l_dim_id         = m_dim_ids_right[l_di];
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

  // determine blocking type
  BinaryPrimitives l_bin_prim;
  l_bin_prim.init( m_dtype_comp,
                   backend_t::BLAS );

  primblo_t l_primitive_blocking = primblo_t::UNDEFINED_PRIMBLO;

  if( m_dim_types_out[ m_num_dims_out - 1 ] == M ) {
    l_primitive_blocking = primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB;
  }
  else if( m_dim_types_out[ m_num_dims_out - 1 ] == C ) {
    l_primitive_blocking = primblo_t::LEFT_X_CB_KB_MB_RIGHT_X_CB_NB_KB_OUT_NB_X_MB_CB;
  }
  else {
    return einsum_ir::COMPILATION_FAILED;
  }

  // perform blocking
  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  l_err = l_bin_prim.blocking( l_primitive_blocking,
                               m_num_dims_left,
                               m_num_dims_right,
                               m_num_dims_out,
                               m_dim_ids_left,
                               m_dim_ids_right,
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

  //determine loop execution order
  if( m_loop_ids_ext == nullptr ){
    m_loop_ids_int.clear();
    m_loop_ids_int.reserve( l_dim_ids_bc.size() + l_dim_ids_bm.size() + l_dim_ids_bn.size() + l_dim_ids_bk.size() );
    m_loop_ids_int.insert( m_loop_ids_int.end(), l_dim_ids_bc.begin(), l_dim_ids_bc.end() );
    m_loop_ids_int.insert( m_loop_ids_int.end(), l_dim_ids_bn.begin(), l_dim_ids_bn.end() );
    m_loop_ids_int.insert( m_loop_ids_int.end(), l_dim_ids_bm.begin(), l_dim_ids_bm.end() );
    m_loop_ids_int.insert( m_loop_ids_int.end(), l_dim_ids_bk.begin(), l_dim_ids_bk.end() );
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

  // derive strides
  std::map< int64_t, int64_t > l_strides_left;
  std::map< int64_t, int64_t > l_strides_right;
  std::map< int64_t, int64_t > l_strides_out;

  strides( m_num_dims_left,
           m_dim_ids_left,
           m_dim_sizes_inner,
           &l_strides_left );

  strides( m_num_dims_right,
           m_dim_ids_right,
           m_dim_sizes_inner,
           &l_strides_right );

  strides( m_num_dims_out,
           m_dim_ids_out,
           m_dim_sizes_inner,
           &l_strides_out );

  // check that no I or J dimensions are present
  if( m_num_dims_i > 0 || m_num_dims_j > 0 ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // derive BLAS parameters
  int64_t l_blas_size_c = 1;
  int64_t l_blas_size_m = 1;
  int64_t l_blas_size_n = 1;
  int64_t l_blas_size_k = 1;

  for( std::size_t l_cb = 0; l_cb < l_dim_ids_cb.size(); l_cb++ ) {
    int64_t l_dim_id = l_dim_ids_cb[l_cb];
    l_blas_size_c *= m_dim_sizes_inner->at( l_dim_id );
  }
  for( std::size_t l_mb = 0; l_mb < l_dim_ids_mb.size(); l_mb++ ) {
    int64_t l_dim_id = l_dim_ids_mb[l_mb];
    l_blas_size_m *= m_dim_sizes_inner->at( l_dim_id );
  }
  for( std::size_t l_nb = 0; l_nb < l_dim_ids_nb.size(); l_nb++ ) {
    int64_t l_dim_id = l_dim_ids_nb[l_nb];
    l_blas_size_n *= m_dim_sizes_inner->at( l_dim_id );
  }
  for( std::size_t l_kb = 0; l_kb < l_dim_ids_kb.size(); l_kb++ ) {
    int64_t l_dim_id = l_dim_ids_kb[l_kb];
    l_blas_size_k *= m_dim_sizes_inner->at( l_dim_id );
  }

  // set leading dimensions
  int64_t l_blas_ld_a = l_dim_ids_kb.size() > 0 ? l_strides_left.at(  l_dim_ids_kb.back() ) : l_blas_size_m;
  int64_t l_blas_ld_b = l_dim_ids_nb.size() > 0 ? l_strides_right.at( l_dim_ids_nb.back() ) : l_blas_size_n;
  int64_t l_blas_ld_c = l_dim_ids_nb.size() > 0 ? l_strides_out.at(   l_dim_ids_nb.back() ) : l_blas_size_c * l_blas_size_m;

  // check that the same data type is used everywhere
  if(    m_dtype_comp != m_dtype_left
      || m_dtype_comp != m_dtype_right
      || m_dtype_comp != m_dtype_out ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }

  // check supported kernel types
  if(    m_ktype_first_touch != einsum_ir::kernel_t::UNDEFINED_KTYPE
      && m_ktype_first_touch != einsum_ir::kernel_t::ZERO
      && m_ktype_first_touch != einsum_ir::kernel_t::CPX_ZERO ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }
  if(    m_ktype_main != einsum_ir::kernel_t::MADD
      && m_ktype_main != einsum_ir::kernel_t::CPX_MADD ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }
  if( m_ktype_last_touch != einsum_ir::kernel_t::UNDEFINED_KTYPE ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }

  // init contraction loops
  m_cont_loops.init( m_dim_sizes_inner,
                     &l_strides_left,
                     &l_strides_right,
                     &l_strides_out,
                     &l_strides_out,
                     &m_dim_types,
                     &m_loop_ids_int,
                     m_dtype_comp,
                     false,
                     false,
                     l_blas_size_c,
                     l_blas_size_m,
                     l_blas_size_n,
                     l_blas_size_k,
                     l_blas_ld_a,
                     l_blas_ld_b,
                     l_blas_ld_c,
                     m_ktype_first_touch,
                     m_ktype_main,
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
  m_cont_loops.contract( i_tensor_left,
                         i_tensor_right,
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