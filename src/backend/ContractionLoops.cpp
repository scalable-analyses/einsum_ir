#include "ContractionLoops.h"

void einsum_ir::backend::ContractionLoops::init( int64_t         i_num_dims_c,
                                                 int64_t         i_num_dims_m,
                                                 int64_t         i_num_dims_n,
                                                 int64_t         i_num_dims_k,
                                                 int64_t const * i_sizes_c,
                                                 int64_t const * i_sizes_m,
                                                 int64_t const * i_sizes_n,
                                                 int64_t const * i_sizes_k,
                                                 int64_t const * i_strides_in_left_c,
                                                 int64_t const * i_strides_in_left_m,
                                                 int64_t const * i_strides_in_left_k,
                                                 int64_t const * i_strides_in_right_c,
                                                 int64_t const * i_strides_in_right_n,
                                                 int64_t const * i_strides_in_right_k,
                                                 int64_t const * i_strides_out_aux_c,
                                                 int64_t const * i_strides_out_aux_m,
                                                 int64_t const * i_strides_out_aux_n,
                                                 int64_t const * i_strides_out_c,
                                                 int64_t const * i_strides_out_m,
                                                 int64_t const * i_strides_out_n,
                                                 int64_t         i_num_bytes_scalar_left,
                                                 int64_t         i_num_bytes_scalar_right,
                                                 int64_t         i_num_bytes_scalar_out ) {
  m_num_dims_c = i_num_dims_c;
  m_num_dims_m = i_num_dims_m;
  m_num_dims_n = i_num_dims_n;
  m_num_dims_k = i_num_dims_k;

  m_sizes_c = i_sizes_c;
  m_sizes_m = i_sizes_m;
  m_sizes_n = i_sizes_n;
  m_sizes_k = i_sizes_k;

  m_strides_in_left_c = i_strides_in_left_c;
  m_strides_in_left_m = i_strides_in_left_m;
  m_strides_in_left_k = i_strides_in_left_k;

  m_strides_in_right_c = i_strides_in_right_c;
  m_strides_in_right_n = i_strides_in_right_n;
  m_strides_in_right_k = i_strides_in_right_k;

  m_strides_out_aux_c = i_strides_out_aux_c;
  m_strides_out_aux_m = i_strides_out_aux_m;
  m_strides_out_aux_n = i_strides_out_aux_n;

  m_strides_out_c = i_strides_out_c;
  m_strides_out_m = i_strides_out_m;
  m_strides_out_n = i_strides_out_n;

  m_num_bytes_scalar_left = i_num_bytes_scalar_left;
  m_num_bytes_scalar_right = i_num_bytes_scalar_right;
  m_num_bytes_scalar_out = i_num_bytes_scalar_out;

  m_threading_num_loops = -1;

  m_threading_first_last_touch = false;

  m_compiled = false;
}

einsum_ir::err_t einsum_ir::backend::ContractionLoops::compile() {
  m_num_loops = m_num_dims_c + m_num_dims_m + m_num_dims_n + m_num_dims_k;

  m_loop_first_last_touch.clear();
  m_loop_dim_type.clear();
  m_loop_sizes.clear();
  m_loop_strides_left.clear();
  m_loop_strides_right.clear();
  m_loop_strides_out_aux.clear();
  m_loop_strides_out.clear();

  m_loop_first_last_touch.reserve( m_num_loops );
  m_loop_dim_type.reserve(         m_num_loops );
  m_loop_sizes.reserve(            m_num_loops );
  m_loop_strides_left.reserve(     m_num_loops );
  m_loop_strides_right.reserve(    m_num_loops );
  m_loop_strides_out_aux.reserve(  m_num_loops );
  m_loop_strides_out.reserve(      m_num_loops );

  // add data of C dimensions
  for( int64_t l_di = 0; l_di < m_num_dims_c; l_di++ ) {
    if( m_num_dims_k == 0 && m_num_dims_m == 0 && m_num_dims_n == 0 ) {
      m_loop_first_last_touch.push_back( EVERY_ITER );
    }
    else {
      m_loop_first_last_touch.push_back( NONE );
    }
    m_loop_dim_type.push_back(        dim_t::C                   );
    m_loop_sizes.push_back(           m_sizes_c[l_di]            );
    m_loop_strides_left.push_back(    m_strides_in_left_c[l_di]  );
    m_loop_strides_right.push_back(   m_strides_in_right_c[l_di] );
    m_loop_strides_out_aux.push_back( m_strides_out_aux_c[l_di]  );
    m_loop_strides_out.push_back(     m_strides_out_c[l_di]      );
  }

  // add data of N dimensions
  for( int64_t l_di = 0; l_di < m_num_dims_n; l_di++ ) {
    if( m_num_dims_k == 0 && m_num_dims_m == 0 ) {
      m_loop_first_last_touch.push_back( EVERY_ITER );
    }
    else {
      m_loop_first_last_touch.push_back( NONE );
    }
    m_loop_dim_type.push_back(        dim_t::N                   );
    m_loop_sizes.push_back(           m_sizes_n[l_di]            );
    m_loop_strides_left.push_back(    0                          );
    m_loop_strides_right.push_back(   m_strides_in_right_n[l_di] );
    m_loop_strides_out_aux.push_back( m_strides_out_aux_n[l_di]  );
    m_loop_strides_out.push_back(     m_strides_out_n[l_di]      );
  }

  // add data of M dimensions
  for( int64_t l_di = 0; l_di < m_num_dims_m; l_di++ ) {
    if( m_num_dims_k == 0 ) {
      m_loop_first_last_touch.push_back( EVERY_ITER );
    }
    else {
      m_loop_first_last_touch.push_back( NONE );
    }
    m_loop_dim_type.push_back(        dim_t::M                  );
    m_loop_sizes.push_back(           m_sizes_m[l_di]           );
    m_loop_strides_left.push_back(    m_strides_in_left_m[l_di] );
    m_loop_strides_right.push_back(   0                         );
    m_loop_strides_out_aux.push_back( m_strides_out_aux_m[l_di] );
    m_loop_strides_out.push_back(     m_strides_out_m[l_di]     );
  }

  // add data of K dimensions
  for( int64_t l_di = 0; l_di < m_num_dims_k; l_di++ ) {
    if( l_di == 0 ) {
      m_loop_first_last_touch.push_back( BEFORE_AFTER_ITER );
    }
    else {
      m_loop_first_last_touch.push_back( NONE );
    }
    m_loop_dim_type.push_back(        dim_t::K                   );
    m_loop_sizes.push_back(           m_sizes_k[l_di]            );
    m_loop_strides_left.push_back(    m_strides_in_left_k[l_di]  );
    m_loop_strides_right.push_back(   m_strides_in_right_k[l_di] );
    m_loop_strides_out_aux.push_back( 0                          );
    m_loop_strides_out.push_back(     0                          );
  }

  // add dummy data for non-existing loops such that the inner kernel is executed
  if( m_num_loops == 0 ) {
    m_loop_dim_type.push_back(         dim_t::UNDEFINED_DIM );
    m_loop_first_last_touch.push_back( EVERY_ITER           );
    m_loop_sizes.push_back(            1                    );
    m_loop_strides_left.push_back(     0                    );
    m_loop_strides_right.push_back(    0                    );
    m_loop_strides_out_aux.push_back(  0                    );
    m_loop_strides_out.push_back(      0                    );
  }

  // scale with size of data types
  for( int64_t l_lo = 0; l_lo < m_num_loops; l_lo++ ) {
    m_loop_strides_left[l_lo]    *= m_num_bytes_scalar_left;
    m_loop_strides_right[l_lo]   *= m_num_bytes_scalar_right;
    m_loop_strides_out_aux[l_lo] *= m_num_bytes_scalar_out;
    m_loop_strides_out[l_lo]     *= m_num_bytes_scalar_out;
  }

  m_compiled = true;

  return einsum_ir::SUCCESS;
}

void einsum_ir::backend::ContractionLoops::threading( int64_t i_num_tasks_target ) {
  if( i_num_tasks_target < 2 ) {
    m_threading_num_loops = -1;
    return;
  }

  m_threading_first_last_touch = false;
  int64_t l_num_tasks = 1;

  for( int64_t l_lo = 0; l_lo < m_num_loops; l_lo++ ) {
    if(    m_threading_first_last_touch  == false
        && m_loop_dim_type[l_lo]         != K
        && m_loop_first_last_touch[l_lo] == NONE ) {
      l_num_tasks *= m_loop_sizes[l_lo];
      m_threading_num_loops = l_lo+1;
    }
    else if(    m_loop_dim_type[l_lo] != K
             && l_lo                  <  2 ) {
      m_threading_first_last_touch = true;
      l_num_tasks *= m_loop_sizes[l_lo];
      m_threading_num_loops = l_lo+1;
    }
    else {
      break;
    }

    if( l_num_tasks >= i_num_tasks_target ) {
      break;
    }
  }
}

void einsum_ir::backend::ContractionLoops::contract_iter( int64_t         i_id_loop,
                                                          void    const * i_ptr_left,
                                                          void    const * i_ptr_right,
                                                          void    const * i_ptr_out_aux,
                                                          void          * i_ptr_out ) {
  // execute first touch kernel
  if( m_loop_first_last_touch[i_id_loop] == BEFORE_AFTER_ITER ) {
    kernel_first_touch( i_ptr_out_aux,
                        i_ptr_out );
  }

  // issue loop iterations
  for( int64_t l_it = 0; l_it < m_loop_sizes[i_id_loop]; l_it++ ) {
    char * l_ptr_left    = (char *) i_ptr_left;
    char * l_ptr_right   = (char *) i_ptr_right;
    char * l_ptr_out_aux = (char *) i_ptr_out_aux;
    char * l_ptr_out     = (char *) i_ptr_out;

    l_ptr_left    += l_it * m_loop_strides_left[    i_id_loop ];
    l_ptr_right   += l_it * m_loop_strides_right[   i_id_loop ];
    l_ptr_out_aux += l_it * m_loop_strides_out_aux[ i_id_loop ];
    l_ptr_out     += l_it * m_loop_strides_out[     i_id_loop ];

    if( m_loop_first_last_touch[i_id_loop] == EVERY_ITER ) {
      kernel_first_touch( l_ptr_out_aux,
                          l_ptr_out );
    }

    if( i_id_loop + 1 < m_num_loops ) {
      contract_iter( i_id_loop+1,
                     l_ptr_left,
                     l_ptr_right,
                     l_ptr_out_aux,
                     l_ptr_out );
    }
    else {
      // execute main kernel
      kernel_main( l_ptr_left,
                   l_ptr_right,
                   l_ptr_out );
    }

    if( m_loop_first_last_touch[i_id_loop] == EVERY_ITER ) {
      kernel_last_touch( l_ptr_out_aux,
                         l_ptr_out );
    }
  }

  // execute last touch kernel
  if( m_loop_first_last_touch[i_id_loop] == BEFORE_AFTER_ITER ) {
    kernel_last_touch( i_ptr_out_aux,
                       i_ptr_out );
  }
}

void einsum_ir::backend::ContractionLoops::contract_iter_parallel_touch_1( int64_t         i_id_loop,
                                                                           void    const * i_ptr_left,
                                                                           void    const * i_ptr_right,
                                                                           void    const * i_ptr_out_aux,
                                                                           void          * i_ptr_out ) {
  // execute first touch kernel
  if( m_loop_first_last_touch[i_id_loop] == BEFORE_AFTER_ITER ) {
    kernel_first_touch( i_ptr_out_aux,
                        i_ptr_out );
  }

  // issue loop iterations
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for( int64_t l_it = 0; l_it < m_loop_sizes[i_id_loop]; l_it++ ) {
    char * l_ptr_left    = (char *) i_ptr_left;
    char * l_ptr_right   = (char *) i_ptr_right;
    char * l_ptr_out_aux = (char *) i_ptr_out_aux;
    char * l_ptr_out     = (char *) i_ptr_out;

    l_ptr_left    += l_it * m_loop_strides_left[    i_id_loop ];
    l_ptr_right   += l_it * m_loop_strides_right[   i_id_loop ];
    l_ptr_out_aux += l_it * m_loop_strides_out_aux[ i_id_loop ];
    l_ptr_out     += l_it * m_loop_strides_out[     i_id_loop ];

    if( m_loop_first_last_touch[i_id_loop] == EVERY_ITER ) {
      kernel_first_touch( l_ptr_out_aux,
                          l_ptr_out );
    }

    if( i_id_loop + 1 < m_num_loops ) {
      contract_iter( i_id_loop+1,
                     l_ptr_left,
                     l_ptr_right,
                     l_ptr_out_aux,
                     l_ptr_out );
    }
    else {
      // execute main kernel
      kernel_main( l_ptr_left,
                   l_ptr_right,
                   l_ptr_out );
    }

    if( m_loop_first_last_touch[i_id_loop] == EVERY_ITER ) {
      kernel_last_touch( l_ptr_out_aux,
                         l_ptr_out );
    }
  }

  // execute last touch kernel
  if( m_loop_first_last_touch[i_id_loop] == BEFORE_AFTER_ITER ) {
    kernel_last_touch( i_ptr_out_aux,
                       i_ptr_out );
  }
}

void einsum_ir::backend::ContractionLoops::contract_iter_parallel_touch_2( int64_t         i_id_loop,
                                                                           void    const * i_ptr_left,
                                                                           void    const * i_ptr_right,
                                                                           void    const * i_ptr_out_aux,
                                                                           void          * i_ptr_out ) {
  // execute first touch kernel
  if( m_loop_first_last_touch[i_id_loop] == BEFORE_AFTER_ITER ) {
    kernel_first_touch( i_ptr_out_aux,
                        i_ptr_out );
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  // issue loop iterations
  for( int64_t l_it_0 = 0; l_it_0 < m_loop_sizes[i_id_loop]; l_it_0++ ) {
    for( int64_t l_it_1 = 0; l_it_1 < m_loop_sizes[i_id_loop+1]; l_it_1++ ) {
      char * l_ptr_left_0    = (char *) i_ptr_left;
      char * l_ptr_right_0   = (char *) i_ptr_right;
      char * l_ptr_out_aux_0 = (char *) i_ptr_out_aux;
      char * l_ptr_out_0     = (char *) i_ptr_out;

      l_ptr_left_0    += l_it_0 * m_loop_strides_left[    i_id_loop ];
      l_ptr_right_0   += l_it_0 * m_loop_strides_right[   i_id_loop ];
      l_ptr_out_aux_0 += l_it_0 * m_loop_strides_out_aux[ i_id_loop ];
      l_ptr_out_0     += l_it_0 * m_loop_strides_out[     i_id_loop ];

      if( m_loop_first_last_touch[i_id_loop] == EVERY_ITER ) {
        kernel_first_touch( l_ptr_out_aux_0,
                            l_ptr_out_0 );
      }

      char * l_ptr_left_1    = l_ptr_left_0    + l_it_1 * m_loop_strides_left[    i_id_loop+1 ];
      char * l_ptr_right_1   = l_ptr_right_0   + l_it_1 * m_loop_strides_right[   i_id_loop+1 ];
      char * l_ptr_out_aux_1 = l_ptr_out_aux_0 + l_it_1 * m_loop_strides_out_aux[ i_id_loop+1 ];
      char * l_ptr_out_1     = l_ptr_out_0     + l_it_1 * m_loop_strides_out[     i_id_loop+1 ];

      if( m_loop_first_last_touch[i_id_loop+1] == EVERY_ITER ) {
        kernel_first_touch( l_ptr_out_aux_1,
                            l_ptr_out_1 );
      }

      if( i_id_loop + 2 < m_num_loops ) {
        contract_iter( i_id_loop+2,
                       l_ptr_left_1,
                       l_ptr_right_1,
                       l_ptr_out_aux_1,
                       l_ptr_out_1 );
      }
      else {
        // execute main kernel
        kernel_main( l_ptr_left_1,
                     l_ptr_right_1,
                     l_ptr_out_1 );
      }

      if( m_loop_first_last_touch[i_id_loop] == EVERY_ITER ) {
        kernel_last_touch( l_ptr_out_aux_0,
                           l_ptr_out_0 );
      }

      if( m_loop_first_last_touch[i_id_loop+1] == EVERY_ITER ) {
        kernel_last_touch( l_ptr_out_aux_1,
                           l_ptr_out_1 );
      }
    }
  }

  // execute last touch kernel
  if( m_loop_first_last_touch[i_id_loop] == BEFORE_AFTER_ITER ) {
    kernel_last_touch( i_ptr_out_aux,
                       i_ptr_out );
  }
}

void einsum_ir::backend::ContractionLoops::contract_iter_parallel_1( int64_t         i_id_loop,
                                                                     void    const * i_ptr_left,
                                                                     void    const * i_ptr_right,
                                                                     void    const * i_ptr_out_aux,
                                                                     void          * i_ptr_out ) {
  // issue loop iterations
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for( int64_t l_it = 0; l_it < m_loop_sizes[i_id_loop]; l_it++ ) {
    char * l_ptr_left    = (char *) i_ptr_left;
    char * l_ptr_right   = (char *) i_ptr_right;
    char * l_ptr_out_aux = (char *) i_ptr_out_aux;
    char * l_ptr_out     = (char *) i_ptr_out;

    l_ptr_left    += l_it * m_loop_strides_left[    i_id_loop ];
    l_ptr_right   += l_it * m_loop_strides_right[   i_id_loop ];
    l_ptr_out_aux += l_it * m_loop_strides_out_aux[ i_id_loop ];
    l_ptr_out     += l_it * m_loop_strides_out[     i_id_loop ];

    if( i_id_loop + 1 < m_num_loops ) {
      contract_iter( i_id_loop+1,
                     l_ptr_left,
                     l_ptr_right,
                     l_ptr_out_aux,
                     l_ptr_out );
    }
    else {
      // execute main kernel
      kernel_main( l_ptr_left,
                   l_ptr_right,
                   l_ptr_out );
    }
  }
}

void einsum_ir::backend::ContractionLoops::contract_iter_parallel_2( int64_t         i_id_loop,
                                                                     void    const * i_ptr_left,
                                                                     void    const * i_ptr_right,
                                                                     void    const * i_ptr_out_aux,
                                                                     void          * i_ptr_out ) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  // issue loop iterations
  for( int64_t l_it_0 = 0; l_it_0 < m_loop_sizes[i_id_loop]; l_it_0++ ) {
    for( int64_t l_it_1 = 0; l_it_1 < m_loop_sizes[i_id_loop+1]; l_it_1++ ) {
      char * l_ptr_left    = (char *) i_ptr_left;
      char * l_ptr_right   = (char *) i_ptr_right;
      char * l_ptr_out_aux = (char *) i_ptr_out_aux;
      char * l_ptr_out     = (char *) i_ptr_out;

      l_ptr_left    += l_it_0 * m_loop_strides_left[    i_id_loop   ];
      l_ptr_right   += l_it_0 * m_loop_strides_right[   i_id_loop   ];
      l_ptr_out_aux += l_it_0 * m_loop_strides_out_aux[ i_id_loop   ];
      l_ptr_out     += l_it_0 * m_loop_strides_out[     i_id_loop   ];

      l_ptr_left    += l_it_1 * m_loop_strides_left[    i_id_loop+1 ];
      l_ptr_right   += l_it_1 * m_loop_strides_right[   i_id_loop+1 ];
      l_ptr_out_aux += l_it_1 * m_loop_strides_out_aux[ i_id_loop+1 ];
      l_ptr_out     += l_it_1 * m_loop_strides_out[     i_id_loop+1 ];

      if( i_id_loop + 2 < m_num_loops ) {
        contract_iter( i_id_loop+2,
                       l_ptr_left,
                       l_ptr_right,
                       l_ptr_out_aux,
                       l_ptr_out );
      }
      else {
        // execute main kernel
        kernel_main( l_ptr_left,
                     l_ptr_right,
                     l_ptr_out );
      }
    }
  }
}

void einsum_ir::backend::ContractionLoops::contract_iter_parallel_3( int64_t         i_id_loop,
                                                                     void    const * i_ptr_left,
                                                                     void    const * i_ptr_right,
                                                                     void    const * i_ptr_out_aux,
                                                                     void          * i_ptr_out ) {
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  // issue loop iterations
  for( int64_t l_it_0 = 0; l_it_0 < m_loop_sizes[i_id_loop]; l_it_0++ ) {
    for( int64_t l_it_1 = 0; l_it_1 < m_loop_sizes[i_id_loop+1]; l_it_1++ ) {
      for( int64_t l_it_2 = 0; l_it_2 < m_loop_sizes[i_id_loop+2]; l_it_2++ ) {
        char * l_ptr_left    = (char *) i_ptr_left;
        char * l_ptr_right   = (char *) i_ptr_right;
        char * l_ptr_out_aux = (char *) i_ptr_out_aux;
        char * l_ptr_out     = (char *) i_ptr_out;

        l_ptr_left    += l_it_0 * m_loop_strides_left[    i_id_loop   ];
        l_ptr_right   += l_it_0 * m_loop_strides_right[   i_id_loop   ];
        l_ptr_out_aux += l_it_0 * m_loop_strides_out_aux[ i_id_loop   ];
        l_ptr_out     += l_it_0 * m_loop_strides_out[     i_id_loop   ];

        l_ptr_left    += l_it_1 * m_loop_strides_left[    i_id_loop+1 ];
        l_ptr_right   += l_it_1 * m_loop_strides_right[   i_id_loop+1 ];
        l_ptr_out_aux += l_it_1 * m_loop_strides_out_aux[ i_id_loop+1 ];
        l_ptr_out     += l_it_1 * m_loop_strides_out[     i_id_loop+1 ];

        l_ptr_left    += l_it_2 * m_loop_strides_left[    i_id_loop+2 ];
        l_ptr_right   += l_it_2 * m_loop_strides_right[   i_id_loop+2 ];
        l_ptr_out_aux += l_it_2 * m_loop_strides_out_aux[ i_id_loop+2 ];
        l_ptr_out     += l_it_2 * m_loop_strides_out[     i_id_loop+2 ];

        if( i_id_loop + 3 < m_num_loops ) {
          contract_iter( i_id_loop+3,
                        l_ptr_left,
                        l_ptr_right,
                        l_ptr_out_aux,
                        l_ptr_out );
        }
        else {
          // execute main kernel
          kernel_main( l_ptr_left,
                       l_ptr_right,
                       l_ptr_out );
        }
      }
    }
  }
}

void einsum_ir::backend::ContractionLoops::contract_iter_parallel_4( int64_t         i_id_loop,
                                                                     void    const * i_ptr_left,
                                                                     void    const * i_ptr_right,
                                                                     void    const * i_ptr_out_aux,
                                                                     void          * i_ptr_out ) {
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
  // issue loop iterations
  for( int64_t l_it_0 = 0; l_it_0 < m_loop_sizes[i_id_loop]; l_it_0++ ) {
    for( int64_t l_it_1 = 0; l_it_1 < m_loop_sizes[i_id_loop+1]; l_it_1++ ) {
      for( int64_t l_it_2 = 0; l_it_2 < m_loop_sizes[i_id_loop+2]; l_it_2++ ) {
        for( int64_t l_it_3 = 0; l_it_3 < m_loop_sizes[i_id_loop+3]; l_it_3++ ) {
          char * l_ptr_left    = (char *) i_ptr_left;
          char * l_ptr_right   = (char *) i_ptr_right;
          char * l_ptr_out_aux = (char *) i_ptr_out_aux;
          char * l_ptr_out     = (char *) i_ptr_out;

          l_ptr_left    += l_it_0 * m_loop_strides_left[    i_id_loop   ];
          l_ptr_right   += l_it_0 * m_loop_strides_right[   i_id_loop   ];
          l_ptr_out_aux += l_it_0 * m_loop_strides_out_aux[ i_id_loop   ];
          l_ptr_out     += l_it_0 * m_loop_strides_out[     i_id_loop   ];

          l_ptr_left    += l_it_1 * m_loop_strides_left[    i_id_loop+1 ];
          l_ptr_right   += l_it_1 * m_loop_strides_right[   i_id_loop+1 ];
          l_ptr_out_aux += l_it_1 * m_loop_strides_out_aux[ i_id_loop+1 ];
          l_ptr_out     += l_it_1 * m_loop_strides_out[     i_id_loop+1 ];

          l_ptr_left    += l_it_2 * m_loop_strides_left[    i_id_loop+2 ];
          l_ptr_right   += l_it_2 * m_loop_strides_right[   i_id_loop+2 ];
          l_ptr_out_aux += l_it_2 * m_loop_strides_out_aux[ i_id_loop+2 ];
          l_ptr_out     += l_it_2 * m_loop_strides_out[     i_id_loop+2 ];

          l_ptr_left    += l_it_3 * m_loop_strides_left[    i_id_loop+3 ];
          l_ptr_right   += l_it_3 * m_loop_strides_right[   i_id_loop+3 ];
          l_ptr_out_aux += l_it_3 * m_loop_strides_out_aux[ i_id_loop+3 ];
          l_ptr_out     += l_it_3 * m_loop_strides_out[     i_id_loop+3 ];

          if( i_id_loop + 4 < m_num_loops ) {
            contract_iter( i_id_loop+4,
                          l_ptr_left,
                          l_ptr_right,
                          l_ptr_out_aux,
                          l_ptr_out );
          }
          else {
            // execute main kernel
            kernel_main( l_ptr_left,
                         l_ptr_right,
                         l_ptr_out );
          }
        }
      }
    }
  }
}

void einsum_ir::backend::ContractionLoops::contract( void const * i_tensor_left,
                                                     void const * i_tensor_right,
                                                     void const * i_tensor_out_aux,
                                                     void       * io_tensor_out ) {
  if( m_threading_num_loops < 1 ) {
    contract_iter( 0,
                   i_tensor_left,
                   i_tensor_right,
                   i_tensor_out_aux,
                   io_tensor_out );
  }
  else {
    if( m_threading_num_loops == 1 ) {
      if( m_threading_first_last_touch ) {
        contract_iter_parallel_touch_1( 0,
                                        i_tensor_left,
                                        i_tensor_right,
                                        i_tensor_out_aux,
                                        io_tensor_out );
      }
      else {
        contract_iter_parallel_1( 0,
                                  i_tensor_left,
                                  i_tensor_right,
                                  i_tensor_out_aux,
                                  io_tensor_out );
      }
    }
    else if( m_threading_num_loops == 2 ) {
      if( m_threading_first_last_touch ) {
        contract_iter_parallel_touch_2( 0,
                                        i_tensor_left,
                                        i_tensor_right,
                                        i_tensor_out_aux,
                                        io_tensor_out );
      }
      else {
        contract_iter_parallel_2( 0,
                                  i_tensor_left,
                                  i_tensor_right,
                                  i_tensor_out_aux,
                                  io_tensor_out );
      }
    }
    else if( m_threading_num_loops == 3 ) {
      contract_iter_parallel_3( 0,
                                i_tensor_left,
                                i_tensor_right,
                                i_tensor_out_aux,
                                io_tensor_out );
    }
    else {
      contract_iter_parallel_4( 0,
                                i_tensor_left,
                                i_tensor_right,
                                i_tensor_out_aux,
                                io_tensor_out );
    }
  }
}