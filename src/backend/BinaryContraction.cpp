#include "BinaryContraction.h"
#include <list>
#include <algorithm>
#include <cassert>
#include <cmath>
#include "ContractionLoops.h"

void einsum_ir::backend::BinaryContraction::dim_types( int64_t         i_num_dims_t0,
                                                       int64_t         i_num_dims_t1,
                                                       int64_t         i_num_dims_t2,
                                                       int64_t const * i_dim_ids_t0,
                                                       int64_t const * i_dim_ids_t1,
                                                       int64_t const * i_dim_ids_t2,
                                                       dim_t           i_dim_type_t2,
                                                       dim_t           i_dim_type_t2_t0,
                                                       dim_t           i_dim_type_t2_t1,
                                                       dim_t           i_dim_type_t2_t0_t1,
                                                       dim_t         * o_dim_types_t2 ) {
  std::list< int64_t > l_dim_ids_t0( i_dim_ids_t0,
                                     i_dim_ids_t0 + i_num_dims_t0 );

  std::list< int64_t > l_dim_ids_t1( i_dim_ids_t1,
                                     i_dim_ids_t1 + i_num_dims_t1 );

  std::list< int64_t > l_dim_ids_t2( i_dim_ids_t2,
                                     i_dim_ids_t2 + i_num_dims_t2 );

  int64_t l_pos_t2 = 0;
  while( l_dim_ids_t2.size() > 0 ) {
    int64_t l_id_t2 = l_dim_ids_t2.front();
    l_dim_ids_t2.pop_front();

    std::list< int64_t >::iterator l_iter_t0 = std::find( l_dim_ids_t0.begin(),
                                                          l_dim_ids_t0.end(),
                                                          l_id_t2 );

    std::list< int64_t >::iterator l_iter_t1 = std::find( l_dim_ids_t1.begin(),
                                                          l_dim_ids_t1.end(),
                                                          l_id_t2 );

    if(         l_iter_t0 == l_dim_ids_t0.end()
             && l_iter_t1 == l_dim_ids_t1.end() ) {
      o_dim_types_t2[l_pos_t2] = i_dim_type_t2;
    }
    else if(    l_iter_t0 != l_dim_ids_t0.end()
             && l_iter_t1 == l_dim_ids_t1.end() ) {
      o_dim_types_t2[l_pos_t2] = i_dim_type_t2_t0;
      l_dim_ids_t0.erase( l_iter_t0 );
    }
    else if(    l_iter_t0 == l_dim_ids_t0.end()
             && l_iter_t1 != l_dim_ids_t1.end() ) {
      o_dim_types_t2[l_pos_t2] = i_dim_type_t2_t1;
      l_dim_ids_t0.erase( l_iter_t1 );
    }
    else {
       o_dim_types_t2[l_pos_t2] = i_dim_type_t2_t0_t1;
       l_dim_ids_t0.erase( l_iter_t0 );
       l_dim_ids_t1.erase( l_iter_t1 );
    }
    l_pos_t2++;
  }
}

int64_t einsum_ir::backend::BinaryContraction::filter_dim_ids( int64_t         i_num_dims_tensor,
                                                               dim_t           i_dim_type_filter,
                                                               dim_t   const * i_dim_types_tensor,
                                                               int64_t const * i_dim_ids_tensor,
                                                               int64_t       * o_dim_ids_filtered ) {
  int64_t l_id_filtered = 0;

  for( int64_t l_di = 0; l_di < i_num_dims_tensor; l_di++ ) {
    if( i_dim_types_tensor[l_di] == i_dim_type_filter ) {
      o_dim_ids_filtered[l_id_filtered] = i_dim_ids_tensor[l_di];
      l_id_filtered++;
    }
  }

  return l_id_filtered;
}

einsum_ir::err_t einsum_ir::backend::BinaryContraction::order_dims_in( tenord_t        i_tensor_ordering,
                                                                        int64_t         i_num_dims_c,
                                                                        int64_t         i_num_dims_m,
                                                                        int64_t         i_num_dims_n,
                                                                        int64_t         i_num_dims_k,
                                                                        int64_t         i_num_dims_cb,
                                                                        int64_t         i_num_dims_mb,
                                                                        int64_t         i_num_dims_nb,
                                                                        int64_t         i_num_dims_kb,
                                                                        int64_t const * i_dim_ids_c,
                                                                        int64_t const * i_dim_ids_m,
                                                                        int64_t const * i_dim_ids_n,
                                                                        int64_t const * i_dim_ids_k,
                                                                        int64_t       * o_dim_ids_left,
                                                                        int64_t       * o_dim_ids_right ) {
  // check for valid input
  if(    i_num_dims_cb > i_num_dims_c
      || i_num_dims_mb > i_num_dims_m
      || i_num_dims_nb > i_num_dims_n
      || i_num_dims_kb > i_num_dims_k ) {
    return DIMENSION_ORDERING_FAILED;
  }

  int64_t l_id_left  = i_num_dims_c + i_num_dims_m + i_num_dims_k - 1;
  int64_t l_id_right = i_num_dims_c + i_num_dims_n + i_num_dims_k - 1;

  int64_t l_id_c = i_num_dims_c - 1;
  int64_t l_id_m = i_num_dims_m - 1;
  int64_t l_id_n = i_num_dims_n - 1;
  int64_t l_id_k = i_num_dims_k - 1;

  int64_t l_num_dims_cb = i_num_dims_cb;
  int64_t l_num_dims_mb = i_num_dims_mb;
  int64_t l_num_dims_nb = i_num_dims_nb;
  int64_t l_num_dims_kb = i_num_dims_kb;

  // add blocking dimensions
  if( i_tensor_ordering == LEFT_BC_BM_BK_RIGHT_BC_BN_BK_OUT_NATIVE ) {
    if(    l_num_dims_cb != 0
        || l_num_dims_mb != 0
        || l_num_dims_nb != 0
        || l_num_dims_kb != 0 ) {
      return DIMENSION_ORDERING_FAILED;
    }
  }
  else if( i_tensor_ordering == LEFT_BC_BM_BK_KB_MB_RIGHT_BC_BN_BK_NB_KB_OUT_NATIVE ) {
    if( l_num_dims_cb != 0 ) {
      return DIMENSION_ORDERING_FAILED;
    }

    while( l_num_dims_mb > 0 ) {
      o_dim_ids_left[l_id_left] = i_dim_ids_m[l_id_m];
      l_id_left--;
      l_id_m--;
      l_num_dims_mb--;
    }

    while( l_num_dims_kb > 0 ) {
      o_dim_ids_left[l_id_left] = i_dim_ids_k[l_id_k];
      o_dim_ids_right[l_id_right] = i_dim_ids_k[l_id_k];
      l_id_left--;
      l_id_right--;
      l_id_k--;
      l_num_dims_kb--;
    }

     while( l_num_dims_nb > 0 ) {
      o_dim_ids_right[l_id_right] = i_dim_ids_n[l_id_n];
      l_id_right--;
      l_id_n--;
      l_num_dims_nb--;
    }
  }
  else if( i_tensor_ordering == LEFT_BC_BM_BK_KB_MB_CB_RIGHT_BC_BN_BK_NB_KB_CB_OUT_NATIVE ) {
    while( l_num_dims_cb > 0 ) {
      o_dim_ids_left[l_id_left] = i_dim_ids_c[l_id_c];
      o_dim_ids_right[l_id_right] = i_dim_ids_c[l_id_c];
      l_id_left--;
      l_id_right--;
      l_id_c--;
      l_num_dims_cb--;
    }

    while( l_num_dims_mb > 0 ) {
      o_dim_ids_left[l_id_left] = i_dim_ids_m[l_id_m];
      l_id_left--;
      l_id_m--;
      l_num_dims_mb--;
    }

    while( l_num_dims_kb > 0 ) {
      o_dim_ids_left[l_id_left] = i_dim_ids_k[l_id_k];
      o_dim_ids_right[l_id_right] = i_dim_ids_k[l_id_k];
      l_id_left--;
      l_id_right--;
      l_id_k--;
      l_num_dims_kb--;
    }

     while( l_num_dims_nb > 0 ) {
      o_dim_ids_right[l_id_right] = i_dim_ids_n[l_id_n];
      l_id_right--;
      l_id_n--;
      l_num_dims_nb--;
    }
  }
  else {
    return einsum_ir::DIMENSION_ORDERING_FAILED;
  }

  // add loop dimensions
  while( l_id_k >= 0 ) {
    o_dim_ids_left[l_id_left]   = i_dim_ids_k[l_id_k];
    o_dim_ids_right[l_id_right] = i_dim_ids_k[l_id_k];

    l_id_left--;
    l_id_right--;
    l_id_k--;
  }

  while( l_id_m >= 0 ) {
    o_dim_ids_left[l_id_left]   = i_dim_ids_m[l_id_m];

    l_id_left--;
    l_id_m--;
  }

  while( l_id_n >= 0 ) {
    o_dim_ids_right[l_id_right] = i_dim_ids_n[l_id_n];

    l_id_right--;
    l_id_n--;
  }

  while( l_id_c >= 0 ) {
    o_dim_ids_left[l_id_left]   = i_dim_ids_c[l_id_c];
    o_dim_ids_right[l_id_right] = i_dim_ids_c[l_id_c];

    l_id_left--;
    l_id_right--;
    l_id_c--;
  }

  return einsum_ir::SUCCESS;
}

void einsum_ir::backend::BinaryContraction::strides( int64_t                              i_num_dims,
                                                     int64_t const *                      i_dim_ids,
                                                     std::map< int64_t, int64_t > const & i_dim_sizes,
                                                     std::map< int64_t, int64_t >       & o_strides ) {
  o_strides.clear();

  int64_t l_id_tensor = i_num_dims - 1;

  int64_t l_stride_tmp = 1;

  while( l_id_tensor >= 0 ) {
    int64_t l_dim_id = i_dim_ids[l_id_tensor];
    int64_t l_dim_size = i_dim_sizes.at( l_dim_id );

    std::pair< int64_t, int64_t > l_pair( l_dim_id,
                                          l_stride_tmp );
    o_strides.insert( l_pair );

    l_stride_tmp *= l_dim_size;

    l_id_tensor--;
  }
}

void einsum_ir::backend::BinaryContraction::strides( int64_t                              i_num_dims_tensor,
                                                     int64_t                              i_num_dims_c,
                                                     int64_t                              i_num_dims_m,
                                                     int64_t                              i_num_dims_n,
                                                     int64_t                              i_num_dims_k,
                                                     int64_t                      const * i_dim_ids_tensor,
                                                     std::map< int64_t, int64_t > const & i_dim_sizes,
                                                     std::map< int64_t, dim_t >   const & i_dim_types,
                                                     int64_t                            * o_strides_c,
                                                     int64_t                            * o_strides_m,
                                                     int64_t                            * o_strides_n,
                                                     int64_t                            * o_strides_k ) {
  int64_t l_id_tensor = i_num_dims_tensor - 1;

  int64_t l_id_c = i_num_dims_c - 1;
  int64_t l_id_m = i_num_dims_m - 1;
  int64_t l_id_n = i_num_dims_n - 1;
  int64_t l_id_k = i_num_dims_k - 1;

  int64_t l_stride_tmp = 1;

  while( l_id_tensor >= 0 ) {
    int64_t l_dim_id = i_dim_ids_tensor[l_id_tensor];
    int64_t l_dim_size = i_dim_sizes.at( l_dim_id );
    dim_t l_dim_type = i_dim_types.at( l_dim_id );

    if(      l_dim_type == C ) {
      o_strides_c[l_id_c] = l_stride_tmp;
      l_id_c--;
    }
    else if( l_dim_type == M ) {
      o_strides_m[l_id_m] = l_stride_tmp;
      l_id_m--;
    }
    else if( l_dim_type == N ) {
      o_strides_n[l_id_n] = l_stride_tmp;
      l_id_n--;
    }
    else if( l_dim_type == K ) {
      o_strides_k[l_id_k] = l_stride_tmp;
      l_id_k--;
    }
    else {
      assert( false );
    }

    l_stride_tmp *= l_dim_size;
    l_id_tensor--;
  }
}

void einsum_ir::backend::BinaryContraction::init( int64_t                              i_num_dims_in_left,
                                                  int64_t                              i_num_dims_in_right,
                                                  int64_t                              i_num_dims_out,
                                                  std::map< int64_t, int64_t > const & i_dim_sizes,
                                                  int64_t                      const * i_dim_ids_in_left,
                                                  int64_t                      const * i_dim_ids_in_right,
                                                  int64_t                      const * i_dim_ids_out,
                                                  data_t                               i_dtype_left,
                                                  data_t                               i_dtype_right,
                                                  data_t                               i_dtype_comp,
                                                  data_t                               i_dtype_out,
                                                  kernel_t                             i_ktype_first_touch,
                                                  kernel_t                             i_ktype_inner,
                                                  kernel_t                             i_ktype_last_touch ) {
  m_num_dims_in_left = i_num_dims_in_left;
  m_num_dims_in_right = i_num_dims_in_right;
  m_num_dims_out = i_num_dims_out;

  m_dim_sizes = &i_dim_sizes;

  m_dim_ids_in_left_native = i_dim_ids_in_left;
  m_dim_ids_in_right_native = i_dim_ids_in_right;
  m_dim_ids_out = i_dim_ids_out;

  m_dtype_left = i_dtype_left;
  m_dtype_right = i_dtype_right;
  m_dtype_comp = i_dtype_comp;
  m_dtype_out = i_dtype_out;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_inner = i_ktype_inner;
  m_ktype_last_touch = i_ktype_last_touch;
}

einsum_ir::err_t einsum_ir::backend::BinaryContraction::compile_base() {
  // derive dimension types
  std::vector< dim_t > l_dim_types_in_left( m_num_dims_in_left );
  m_dim_types_out.resize( m_num_dims_out );

  dim_types( m_num_dims_in_right,
             m_num_dims_out,
             m_num_dims_in_left,
             m_dim_ids_in_right_native,
             m_dim_ids_out,
             m_dim_ids_in_left_native,
             I,
             K,
             M,
             C,
             l_dim_types_in_left.data() );

  dim_types( m_num_dims_in_left,
             m_num_dims_in_right,
             m_num_dims_out,
             m_dim_ids_in_left_native,
             m_dim_ids_in_right_native,
             m_dim_ids_out,
             UNDEFINED_DIM,
             M,
             N,
             C,
             m_dim_types_out.data() );

  // count dimensions
  m_num_dims_c = std::count( m_dim_types_out.begin(),
                             m_dim_types_out.end(),
                             C );
  m_num_dims_m = std::count( m_dim_types_out.begin(),
                             m_dim_types_out.end(),
                             M );
  m_num_dims_n = std::count( m_dim_types_out.begin(),
                             m_dim_types_out.end(),
                             N );
  m_num_dims_k = std::count( l_dim_types_in_left.begin(),
                             l_dim_types_in_left.end(),
                             K );

  // filter dim ids
  m_dim_ids_c.resize( m_num_dims_c );
  m_dim_ids_m.resize( m_num_dims_m );
  m_dim_ids_n.resize( m_num_dims_n );
  m_dim_ids_k.resize( m_num_dims_k );

  filter_dim_ids( m_num_dims_out,
                  C,
                  m_dim_types_out.data(),
                  m_dim_ids_out,
                  m_dim_ids_c.data() );

  filter_dim_ids( m_num_dims_out,
                  M,
                  m_dim_types_out.data(),
                  m_dim_ids_out,
                  m_dim_ids_m.data() );

  filter_dim_ids( m_num_dims_out,
                  N,
                  m_dim_types_out.data(),
                  m_dim_ids_out,
                  m_dim_ids_n.data() );

  filter_dim_ids( m_num_dims_in_left,
                  K,
                  l_dim_types_in_left.data(),
                  m_dim_ids_in_left_native,
                  m_dim_ids_k.data() );

  // derive sizes
  m_sizes_c.resize( m_num_dims_c );
  m_sizes_m.resize( m_num_dims_m );
  m_sizes_n.resize( m_num_dims_n );
  m_sizes_k.resize( m_num_dims_k );

  for( int64_t l_c = 0; l_c < m_num_dims_c; l_c++ ) {
    int64_t l_id = m_dim_ids_c[l_c];
    m_sizes_c[l_c] = m_dim_sizes->at(l_id);
  }
  for( int64_t l_m = 0; l_m < m_num_dims_m; l_m++ ) {
    int64_t l_id = m_dim_ids_m[l_m];
    m_sizes_m[l_m] = m_dim_sizes->at(l_id);
  }
  for( int64_t l_n = 0; l_n < m_num_dims_n; l_n++ ) {
    int64_t l_id = m_dim_ids_n[l_n];
    m_sizes_n[l_n] = m_dim_sizes->at(l_id);
  }
  for( int64_t l_k = 0; l_k < m_num_dims_k; l_k++ ) {
    int64_t l_id = m_dim_ids_k[l_k];
    m_sizes_k[l_k] = m_dim_sizes->at(l_id);
  }

  return einsum_ir::SUCCESS;
}

int64_t const * einsum_ir::backend::BinaryContraction::dim_ids_in_ordered( int64_t i_side ) {
  int64_t const * l_dim_ids = nullptr;

  int64_t l_side = i_side;
  if( m_tensors_in_swapped ) {
    l_side = (l_side == 0) ? 1 : 0;
  }

  if( l_side == 0 ) {
    if( m_dim_ids_left_ordered.size() > 0 ) {
      l_dim_ids = m_dim_ids_left_ordered.data();
    }
    else {
      l_dim_ids = m_dim_ids_in_left_native;
    }
  }
  else if( l_side == 1 ) {
    if( m_dim_ids_right_ordered.size() > 0 ) {
      l_dim_ids = m_dim_ids_right_ordered.data();
    }
    else {
      l_dim_ids = m_dim_ids_in_right_native;
    }
  }

  return l_dim_ids;
}

int64_t einsum_ir::backend::BinaryContraction::num_ops() {
  int64_t l_size_c = 1;
  int64_t l_size_m = 1;
  int64_t l_size_n = 1;
  int64_t l_size_k = 1;

  for( int64_t l_c = 0; l_c < m_num_dims_c; l_c++ ) {
    l_size_c *= m_sizes_c[l_c];
  }
  for( int64_t l_m = 0; l_m < m_num_dims_m; l_m++ ) {
    l_size_m *= m_sizes_m[l_m];
  }
  for( int64_t l_n = 0; l_n < m_num_dims_n; l_n++ ) {
    l_size_n *= m_sizes_n[l_n];
  }
  for( int64_t l_k = 0; l_k < m_num_dims_k; l_k++ ) {
    l_size_k *= m_sizes_k[l_k];
  }

  int64_t l_num_ops = l_size_c * l_size_m * l_size_n * l_size_k * 2;

  if( m_ktype_first_touch == ZERO ) {
    l_num_ops -= l_size_c * l_size_m * l_size_n;
  }

  return l_num_ops;
}