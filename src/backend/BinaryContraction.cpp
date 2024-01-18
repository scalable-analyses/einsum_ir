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


void einsum_ir::backend::BinaryContraction::dim_types_ids( int64_t                                 i_num_dims_left,
                                                           int64_t                                 i_num_dims_right,
                                                           int64_t                                 i_num_dims_out,
                                                           int64_t                         const * i_dim_ids_left,
                                                           int64_t                         const * i_dim_ids_right,
                                                           int64_t                         const * i_dim_ids_out,
                                                           std::vector< einsum_ir::dim_t >       & o_dim_types_out,
                                                           std::vector<          int64_t >       & o_dim_ids_c,
                                                           std::vector<          int64_t >       & o_dim_ids_m,
                                                           std::vector<          int64_t >       & o_dim_ids_n,
                                                           std::vector<          int64_t >       & o_dim_ids_k,
                                                           std::vector<          int64_t >       & o_dim_ids_i,
                                                           std::vector<          int64_t >       & o_dim_ids_j ) {
  // derive dimension types
  std::vector< einsum_ir::dim_t > l_dim_types_left( i_num_dims_left );
  std::vector< einsum_ir::dim_t > l_dim_types_right( i_num_dims_right );
  o_dim_types_out.resize( i_num_dims_out );

  dim_types( i_num_dims_right,
             i_num_dims_out,
             i_num_dims_left,
             i_dim_ids_right,
             i_dim_ids_out,
             i_dim_ids_left,
             einsum_ir::I,
             einsum_ir::K,
             einsum_ir::M,
             einsum_ir::C,
             l_dim_types_left.data() );

  dim_types( i_num_dims_left,
             i_num_dims_out,
             i_num_dims_right,
             i_dim_ids_left,
             i_dim_ids_out,
             i_dim_ids_right,
             einsum_ir::J,
             einsum_ir::K,
             einsum_ir::N,
             einsum_ir::C,
             l_dim_types_right.data() );

  dim_types( i_num_dims_left,
             i_num_dims_right,
             i_num_dims_out,
             i_dim_ids_left,
             i_dim_ids_right,
             i_dim_ids_out,
             einsum_ir::UNDEFINED_DIM,
             einsum_ir::M,
             einsum_ir::N,
             einsum_ir::C,
             o_dim_types_out.data() );

  // count dimensions
  int64_t l_num_dims_c = std::count( o_dim_types_out.begin(),
                                     o_dim_types_out.end(),
                                     einsum_ir::C );
  int64_t l_num_dims_m = std::count( o_dim_types_out.begin(),
                                     o_dim_types_out.end(),
                                     einsum_ir::M );
  int64_t l_num_dims_n = std::count( o_dim_types_out.begin(),
                                     o_dim_types_out.end(),
                                     einsum_ir::N );
  int64_t l_num_dims_k = std::count( l_dim_types_left.begin(),
                                     l_dim_types_left.end(),
                                     einsum_ir::K );
  int64_t l_num_dims_i = std::count( l_dim_types_left.begin(),
                                     l_dim_types_left.end(),
                                     einsum_ir::I );
  int64_t l_num_dims_j = std::count( l_dim_types_right.begin(),
                                     l_dim_types_right.end(),
                                     einsum_ir::J );

  // filter dim ids
  o_dim_ids_c.resize( l_num_dims_c );
  o_dim_ids_m.resize( l_num_dims_m );
  o_dim_ids_n.resize( l_num_dims_n );
  o_dim_ids_k.resize( l_num_dims_k );
  o_dim_ids_i.resize( l_num_dims_i );
  o_dim_ids_j.resize( l_num_dims_j );

  filter_dim_ids( i_num_dims_out,
                  einsum_ir::C,
                  o_dim_types_out.data(),
                  i_dim_ids_out,
                  o_dim_ids_c.data() );

  filter_dim_ids( i_num_dims_out,
                  einsum_ir::M,
                  o_dim_types_out.data(),
                  i_dim_ids_out,
                  o_dim_ids_m.data() );

  filter_dim_ids( i_num_dims_out,
                  einsum_ir::N,
                  o_dim_types_out.data(),
                  i_dim_ids_out,
                  o_dim_ids_n.data() );

  filter_dim_ids( i_num_dims_left,
                  einsum_ir::K,
                  l_dim_types_left.data(),
                  i_dim_ids_left,
                  o_dim_ids_k.data() );

  filter_dim_ids( i_num_dims_left,
                  einsum_ir::I,
                  l_dim_types_left.data(),
                  i_dim_ids_left,
                  o_dim_ids_i.data() );

  filter_dim_ids( i_num_dims_right,
                  einsum_ir::J,
                  l_dim_types_right.data(),
                  i_dim_ids_right,
                  o_dim_ids_j.data() );
}

einsum_ir::err_t einsum_ir::backend::BinaryContraction::order_dims_in( tenord_t        i_tensor_ordering,
                                                                       int64_t         i_num_dims_c,
                                                                       int64_t         i_num_dims_m,
                                                                       int64_t         i_num_dims_n,
                                                                       int64_t         i_num_dims_k,
                                                                       int64_t         i_num_dims_i,
                                                                       int64_t         i_num_dims_j,
                                                                       int64_t         i_num_dims_cb,
                                                                       int64_t         i_num_dims_mb,
                                                                       int64_t         i_num_dims_nb,
                                                                       int64_t         i_num_dims_kb,
                                                                       int64_t         i_num_dims_ib,
                                                                       int64_t         i_num_dims_jb,
                                                                       int64_t const * i_dim_ids_c,
                                                                       int64_t const * i_dim_ids_m,
                                                                       int64_t const * i_dim_ids_n,
                                                                       int64_t const * i_dim_ids_k,
                                                                       int64_t const * i_dim_ids_i,
                                                                       int64_t const * i_dim_ids_j,
                                                                       int64_t       * o_dim_ids_left,
                                                                       int64_t       * o_dim_ids_right ) {
  // check for valid input
  if(    i_num_dims_cb > i_num_dims_c
      || i_num_dims_mb > i_num_dims_m
      || i_num_dims_nb > i_num_dims_n
      || i_num_dims_kb > i_num_dims_k
      || i_num_dims_ib > i_num_dims_i
      || i_num_dims_jb > i_num_dims_j ) {
    return DIMENSION_ORDERING_FAILED;
  }

  int64_t l_id_left  = i_num_dims_c + i_num_dims_m + i_num_dims_k + i_num_dims_i - 1;
  int64_t l_id_right = i_num_dims_c + i_num_dims_n + i_num_dims_k + i_num_dims_j - 1;

  int64_t l_id_c = i_num_dims_c - 1;
  int64_t l_id_m = i_num_dims_m - 1;
  int64_t l_id_n = i_num_dims_n - 1;
  int64_t l_id_k = i_num_dims_k - 1;
  int64_t l_id_i = i_num_dims_i - 1;
  int64_t l_id_j = i_num_dims_j - 1;

  int64_t l_num_dims_cb = i_num_dims_cb;
  int64_t l_num_dims_mb = i_num_dims_mb;
  int64_t l_num_dims_nb = i_num_dims_nb;
  int64_t l_num_dims_kb = i_num_dims_kb;
  int64_t l_num_dims_ib = i_num_dims_ib;
  int64_t l_num_dims_jb = i_num_dims_jb;

  // add blocking dimensions
  if( i_tensor_ordering == LEFT_BC_BM_BI_BK_RIGHT_BC_BN_BJ_BK_OUT_NATIVE ) {
    if(    l_num_dims_cb != 0
        || l_num_dims_mb != 0
        || l_num_dims_nb != 0
        || l_num_dims_kb != 0
        || l_num_dims_ib != 0
        || l_num_dims_jb != 0 ) {
      return DIMENSION_ORDERING_FAILED;
    }
  }
  else if( i_tensor_ordering == LEFT_BC_BM_BI_BK_IB_KB_MB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_OUT_NATIVE ) {
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

    while( l_num_dims_ib > 0 ) {
      o_dim_ids_left[l_id_left] = i_dim_ids_i[l_id_i];
      l_id_left--;
      l_id_i--;
      l_num_dims_ib--;
    }

    while( l_num_dims_jb > 0 ) {
      o_dim_ids_right[l_id_right] = i_dim_ids_j[l_id_j];
      l_id_right--;
      l_id_j--;
      l_num_dims_jb--;
    }
  }
  else if( i_tensor_ordering == LEFT_BC_BM_BI_BK_IB_KB_MB_CB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_CB_OUT_NATIVE ) {
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

    while( l_num_dims_ib > 0 ) {
      o_dim_ids_left[l_id_left] = i_dim_ids_i[l_id_i];
      l_id_left--;
      l_id_i--;
      l_num_dims_ib--;
    }

    while( l_num_dims_jb > 0 ) {
      o_dim_ids_right[l_id_right] = i_dim_ids_j[l_id_j];
      l_id_right--;
      l_id_j--;
      l_num_dims_jb--;
    }
  }
  else if( i_tensor_ordering == LEFT_BC_BM_BI_BK_CB_KB_MB_RIGHT_BC_BN_BJ_BK_CB_NB_KB_OUT_NATIVE ) {
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

    while( l_num_dims_cb > 0 ) {
      o_dim_ids_left[l_id_left] = i_dim_ids_c[l_id_c];
      o_dim_ids_right[l_id_right] = i_dim_ids_c[l_id_c];
      l_id_left--;
      l_id_right--;
      l_id_c--;
      l_num_dims_cb--;
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

  while( l_id_i >= 0 ) {
    o_dim_ids_left[l_id_left]   = i_dim_ids_i[l_id_i];

    l_id_left--;
    l_id_i--;
  }

  while( l_id_j >= 0 ) {
    o_dim_ids_right[l_id_right] = i_dim_ids_j[l_id_j];

    l_id_right--;
    l_id_j--;
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
                                                     std::map< int64_t, int64_t > const * i_dim_sizes,
                                                     std::map< int64_t, int64_t >       * o_strides ) {
  o_strides->clear();

  int64_t l_id_tensor = i_num_dims - 1;

  int64_t l_stride_tmp = 1;

  while( l_id_tensor >= 0 ) {
    int64_t l_dim_id = i_dim_ids[l_id_tensor];
    int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

    std::pair< int64_t, int64_t > l_pair( l_dim_id,
                                          l_stride_tmp );
    if( l_dim_size < 2 ) {
      l_pair.second = 0;
    }

    o_strides->insert( l_pair );

    l_stride_tmp *= l_dim_size;

    l_id_tensor--;
  }
}

void einsum_ir::backend::BinaryContraction::strides( int64_t                              i_num_dims_left,
                                                     int64_t                              i_num_dims_right,
                                                     int64_t                              i_num_dims_out,
                                                     int64_t                              i_num_dims_c,
                                                     int64_t                              i_num_dims_m,
                                                     int64_t                              i_num_dims_n,
                                                     int64_t                              i_num_dims_k,
                                                     int64_t                              i_num_dims_i,
                                                     int64_t                              i_num_dims_j,
                                                     int64_t                      const * i_dim_ids_left,
                                                     int64_t                      const * i_dim_ids_right,
                                                     int64_t                      const * i_dim_ids_out,
                                                     int64_t                      const * i_dim_ids_c,
                                                     int64_t                      const * i_dim_ids_m,
                                                     int64_t                      const * i_dim_ids_n,
                                                     int64_t                      const * i_dim_ids_k,
                                                     int64_t                      const * i_dim_ids_i,
                                                     int64_t                      const * i_dim_ids_j,
                                                     std::map< int64_t, int64_t > const * i_dim_sizes_left,
                                                     std::map< int64_t, int64_t > const * i_dim_sizes_right,
                                                     std::map< int64_t, int64_t > const * i_dim_sizes_out_aux,
                                                     std::map< int64_t, int64_t > const * i_dim_sizes_out,
                                                     int64_t                            * o_strides_left_c,
                                                     int64_t                            * o_strides_left_m,
                                                     int64_t                            * o_strides_left_k,
                                                     int64_t                            * o_strides_left_i,
                                                     int64_t                            * o_strides_right_c,
                                                     int64_t                            * o_strides_right_n,
                                                     int64_t                            * o_strides_right_k,
                                                     int64_t                            * o_strides_right_j,
                                                     int64_t                            * o_strides_out_aux_c,
                                                     int64_t                            * o_strides_out_aux_m,
                                                     int64_t                            * o_strides_out_aux_n,
                                                     int64_t                            * o_strides_out_c,
                                                     int64_t                            * o_strides_out_m,
                                                     int64_t                            * o_strides_out_n ) {
  std::map< int64_t, int64_t > l_map_id_stride_left;
  std::map< int64_t, int64_t > l_map_id_stride_right;
  std::map< int64_t, int64_t > l_map_id_stride_out_aux;
  std::map< int64_t, int64_t > l_map_id_stride_out;

  strides( i_num_dims_left,
           i_dim_ids_left,
           i_dim_sizes_left,
           &l_map_id_stride_left );

  strides( i_num_dims_right,
           i_dim_ids_right,
           i_dim_sizes_right,
           &l_map_id_stride_right );

  strides( i_num_dims_out,
           i_dim_ids_out,
           i_dim_sizes_out,
           &l_map_id_stride_out );

  if( i_dim_sizes_out_aux != nullptr ) {
    strides( i_num_dims_out,
            i_dim_ids_out,
            i_dim_sizes_out_aux,
            &l_map_id_stride_out_aux );
  }
  else {
    l_map_id_stride_out_aux = l_map_id_stride_out;
  }

  for( int64_t l_c = 0; l_c < i_num_dims_c; l_c++ ) {
    int64_t l_id = i_dim_ids_c[l_c];
    o_strides_left_c[l_c]    = l_map_id_stride_left.at(l_id);
    o_strides_right_c[l_c]   = l_map_id_stride_right.at(l_id);
    o_strides_out_aux_c[l_c] = l_map_id_stride_out_aux.at(l_id);
    o_strides_out_c[l_c]     = l_map_id_stride_out.at(l_id);
  }

  for( int64_t l_m = 0; l_m < i_num_dims_m; l_m++ ) {
    int64_t l_id = i_dim_ids_m[l_m];
    o_strides_left_m[l_m]    = l_map_id_stride_left.at(l_id);
    o_strides_out_aux_m[l_m] = l_map_id_stride_out_aux.at(l_id);
    o_strides_out_m[l_m]     = l_map_id_stride_out.at(l_id);
  }

  for( int64_t l_n = 0; l_n < i_num_dims_n; l_n++ ) {
    int64_t l_id = i_dim_ids_n[l_n];
    o_strides_right_n[l_n]   = l_map_id_stride_right.at(l_id);
    o_strides_out_aux_n[l_n] = l_map_id_stride_out_aux.at(l_id);
    o_strides_out_n[l_n]     = l_map_id_stride_out.at(l_id);
  }

  for( int64_t l_k = 0; l_k < i_num_dims_k; l_k++ ) {
    int64_t l_id = i_dim_ids_k[l_k];
    o_strides_left_k[l_k]  = l_map_id_stride_left.at(l_id);
    o_strides_right_k[l_k] = l_map_id_stride_right.at(l_id);
  }

  for( int64_t l_i = 0; l_i < i_num_dims_i; l_i++ ) {
    int64_t l_id = i_dim_ids_i[l_i];
    o_strides_left_i[l_i]  = l_map_id_stride_left.at(l_id);
  }

  for( int64_t l_j = 0; l_j < i_num_dims_j; l_j++ ) {
    int64_t l_id = i_dim_ids_j[l_j];
    o_strides_right_j[l_j] = l_map_id_stride_right.at(l_id);
  }
}

void einsum_ir::backend::BinaryContraction::apply_stride_multiplier( int64_t         i_num_dims,
                                                                     int64_t const * i_dim_ids,
                                                                     int64_t         i_mult_dim_id,
                                                                     int64_t         i_mult_value,
                                                                     int64_t       * io_strides ) {
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    if( i_dim_ids[l_di] == i_mult_dim_id ) {
      io_strides[l_di] *= i_mult_value;
      break;
    }
  }
}

void einsum_ir::backend::BinaryContraction::apply_stride_multipliers( int64_t                              i_num_dims_c,
                                                                      int64_t                              i_num_dims_m,
                                                                      int64_t                              i_num_dims_n,
                                                                      int64_t                              i_num_dims_k,
                                                                      std::map< int64_t, int64_t > const * i_stride_multipliers_left,
                                                                      std::map< int64_t, int64_t > const * i_stride_multipliers_right,
                                                                      std::map< int64_t, int64_t > const * i_stride_multipliers_out,
                                                                      int64_t                      const * i_dim_ids_c,
                                                                      int64_t                      const * i_dim_ids_m,
                                                                      int64_t                      const * i_dim_ids_n,
                                                                      int64_t                      const * i_dim_ids_k,
                                                                      int64_t                            * io_strides_left_c,
                                                                      int64_t                            * io_strides_left_m,
                                                                      int64_t                            * io_strides_left_k,
                                                                      int64_t                            * io_strides_right_c,
                                                                      int64_t                            * io_strides_right_n,
                                                                      int64_t                            * io_strides_right_k,
                                                                      int64_t                            * io_strides_out_c,
                                                                      int64_t                            * io_strides_out_m,
                                                                      int64_t                            * io_strides_out_n ) {
  if( i_stride_multipliers_left != nullptr ) {
    for( std::pair< int64_t, int64_t > const l_pa : *i_stride_multipliers_left ) {
      // apply to C
      apply_stride_multiplier( i_num_dims_c,
                               i_dim_ids_c,
                               l_pa.first,
                               l_pa.second,
                               io_strides_left_c  );
      // apply to M
      apply_stride_multiplier( i_num_dims_m,
                               i_dim_ids_m,
                               l_pa.first,
                               l_pa.second,
                               io_strides_left_m  );
      // apply to K
      apply_stride_multiplier( i_num_dims_k,
                               i_dim_ids_k,
                               l_pa.first,
                               l_pa.second,
                               io_strides_left_k  );
    }
  }
  if( i_stride_multipliers_right != nullptr ) {
    for( std::pair< int64_t, int64_t > const l_pa : *i_stride_multipliers_right ) {
      // apply to C
      apply_stride_multiplier( i_num_dims_c,
                               i_dim_ids_c,
                               l_pa.first,
                               l_pa.second,
                               io_strides_right_c  );
      // apply to N
      apply_stride_multiplier( i_num_dims_n,
                               i_dim_ids_n,
                               l_pa.first,
                               l_pa.second,
                               io_strides_right_n  );
      // apply to K
      apply_stride_multiplier( i_num_dims_k,
                               i_dim_ids_k,
                               l_pa.first,
                               l_pa.second,
                               io_strides_right_k  );
    }
  }
  if( i_stride_multipliers_out != nullptr ) {
    for( std::pair< int64_t, int64_t > const l_pa : *i_stride_multipliers_out ) {
      // apply to C
      apply_stride_multiplier( i_num_dims_c,
                               i_dim_ids_c,
                               l_pa.first,
                               l_pa.second,
                               io_strides_out_c  );
      // apply to M
      apply_stride_multiplier( i_num_dims_m,
                               i_dim_ids_m,
                               l_pa.first,
                               l_pa.second,
                               io_strides_out_m  );
      // apply to N
      apply_stride_multiplier( i_num_dims_n,
                               i_dim_ids_n,
                               l_pa.first,
                               l_pa.second,
                               io_strides_out_n  );
    }
  }
}

einsum_ir::err_t einsum_ir::backend::BinaryContraction::link_secondary_to_primary( int64_t                              i_dim_id_s,
                                                                                   int64_t                              i_num_dims_p,
                                                                                   int64_t                      const * i_dim_ids_p,
                                                                                   std::map< int64_t, int64_t > const & i_dim_link_s_to_p,
                                                                                   int64_t                            & o_location ) {
  // search for corresponding primary dimension
  std::map< int64_t, int64_t >::const_iterator l_it_primary;
  l_it_primary = i_dim_link_s_to_p.find( i_dim_id_s );

  if( l_it_primary != i_dim_link_s_to_p.end() ) {
    int64_t l_id_primary = l_it_primary->second;

    // determine location among primary dimensions
    int64_t l_di = 0;

    while( l_di < i_num_dims_p ) {
      if( i_dim_ids_p[l_di] == l_id_primary ) {
        break;
      }
      l_di++;
    }

    if(    l_di >= i_num_dims_p
        || i_dim_ids_p[l_di] != l_id_primary ) {
      return err_t::INVALID_ID;
    }

    o_location = l_di;
  }
  else {
    return err_t::INVALID_ID;
  }

  return err_t::SUCCESS;
}

void einsum_ir::backend::BinaryContraction::init( int64_t                              i_num_dims_left,
                                                  int64_t                              i_num_dims_right,
                                                  int64_t                              i_num_dims_out,
                                                  std::map< int64_t, int64_t > const * i_dim_sizes_inner,
                                                  std::map< int64_t, int64_t > const * i_dim_sizes_outer_left,
                                                  std::map< int64_t, int64_t > const * i_dim_sizes_outer_right,
                                                  std::map< int64_t, int64_t > const * i_dim_sizes_outer_out_aux,
                                                  std::map< int64_t, int64_t > const * i_dim_sizes_outer_out,
                                                  std::map< int64_t, int64_t > const * i_stride_multipliers_left,
                                                  std::map< int64_t, int64_t > const * i_stride_multipliers_right,
                                                  std::map< int64_t, int64_t > const * i_stride_multipliers_out,
                                                  int64_t                      const * i_dim_ids_left,
                                                  int64_t                      const * i_dim_ids_right,
                                                  int64_t                      const * i_dim_ids_out,
                                                  std::map< int64_t, int64_t > const * i_dim_link_s_to_p,
                                                  data_t                               i_dtype_left,
                                                  data_t                               i_dtype_right,
                                                  data_t                               i_dtype_comp,
                                                  data_t                               i_dtype_out,
                                                  kernel_t                             i_ktype_first_touch,
                                                  kernel_t                             i_ktype_main,
                                                  kernel_t                             i_ktype_last_touch ) {
  m_num_dims_left   = i_num_dims_left;
  m_num_dims_right = i_num_dims_right;
  m_num_dims_out   = i_num_dims_out;

  m_dim_sizes_inner         = i_dim_sizes_inner;
  m_dim_sizes_outer_left    = i_dim_sizes_outer_left;
  m_dim_sizes_outer_right   = i_dim_sizes_outer_right;
  m_dim_sizes_outer_out_aux = i_dim_sizes_outer_out_aux;
  m_dim_sizes_outer_out     = i_dim_sizes_outer_out;

  m_stride_multipliers_left  = i_stride_multipliers_left;
  m_stride_multipliers_right = i_stride_multipliers_right;
  m_stride_multipliers_out   = i_stride_multipliers_out;

  m_dim_ids_left_native  = i_dim_ids_left;
  m_dim_ids_right_native = i_dim_ids_right;
  m_dim_ids_out          = i_dim_ids_out;

  m_dim_link_s_to_p = i_dim_link_s_to_p;

  m_dtype_left  = i_dtype_left;
  m_dtype_right = i_dtype_right;
  m_dtype_comp  = i_dtype_comp;
  m_dtype_out   = i_dtype_out;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main        = i_ktype_main;
  m_ktype_last_touch  = i_ktype_last_touch;
}

einsum_ir::err_t einsum_ir::backend::BinaryContraction::compile_base() {
  dim_types_ids( m_num_dims_left,
                 m_num_dims_right,
                 m_num_dims_out,
                 m_dim_ids_left_native,
                 m_dim_ids_right_native,
                 m_dim_ids_out,
                 m_dim_types_out,
                 m_dim_ids_c,
                 m_dim_ids_m,
                 m_dim_ids_n,
                 m_dim_ids_k,
                 m_dim_ids_i,
                 m_dim_ids_j );

  m_num_dims_c = m_dim_ids_c.size();
  m_num_dims_m = m_dim_ids_m.size();
  m_num_dims_n = m_dim_ids_n.size();
  m_num_dims_k = m_dim_ids_k.size();
  m_num_dims_i = m_dim_ids_i.size();
  m_num_dims_j = m_dim_ids_j.size();

  // derive sizes
  m_sizes_c.resize( m_num_dims_c );
  m_sizes_m.resize( m_num_dims_m );
  m_sizes_n.resize( m_num_dims_n );
  m_sizes_k.resize( m_num_dims_k );
  m_sizes_i.resize( m_num_dims_i );
  m_sizes_j.resize( m_num_dims_j );

  for( int64_t l_c = 0; l_c < m_num_dims_c; l_c++ ) {
    int64_t l_id = m_dim_ids_c[l_c];
    m_sizes_c[l_c] = m_dim_sizes_inner->at(l_id);
  }
  for( int64_t l_m = 0; l_m < m_num_dims_m; l_m++ ) {
    int64_t l_id = m_dim_ids_m[l_m];
    m_sizes_m[l_m] = m_dim_sizes_inner->at(l_id);
  }
  for( int64_t l_n = 0; l_n < m_num_dims_n; l_n++ ) {
    int64_t l_id = m_dim_ids_n[l_n];
    m_sizes_n[l_n] = m_dim_sizes_inner->at(l_id);
  }
  for( int64_t l_k = 0; l_k < m_num_dims_k; l_k++ ) {
    int64_t l_id = m_dim_ids_k[l_k];
    m_sizes_k[l_k] = m_dim_sizes_inner->at(l_id);
  }
  for( int64_t l_i = 0; l_i < m_num_dims_i; l_i++ ) {
    int64_t l_id = m_dim_ids_i[l_i];
    m_sizes_i[l_i] = m_dim_sizes_inner->at(l_id);
  }
  for( int64_t l_j = 0; l_j < m_num_dims_j; l_j++ ) {
    int64_t l_id = m_dim_ids_j[l_j];
    m_sizes_j[l_j] = m_dim_sizes_inner->at(l_id);
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
      l_dim_ids = m_dim_ids_left_native;
    }
  }
  else if( l_side == 1 ) {
    if( m_dim_ids_right_ordered.size() > 0 ) {
      l_dim_ids = m_dim_ids_right_ordered.data();
    }
    else {
      l_dim_ids = m_dim_ids_right_native;
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