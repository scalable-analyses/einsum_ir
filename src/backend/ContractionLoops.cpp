#include "ContractionLoops.h"
//TODO remove
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::backend::ContractionLoops::init( int64_t                 i_num_dims_c,
                                                 int64_t                 i_num_dims_m,
                                                 int64_t                 i_num_dims_n,
                                                 int64_t                 i_num_dims_k,
                                                 int64_t         const * i_sizes_c,
                                                 int64_t         const * i_sizes_m,
                                                 int64_t         const * i_sizes_n,
                                                 int64_t         const * i_sizes_k,
                                                 int64_t         const * i_strides_in_left_c,
                                                 int64_t         const * i_strides_in_left_m,
                                                 int64_t         const * i_strides_in_left_k,
                                                 int64_t         const * i_strides_in_right_c,
                                                 int64_t         const * i_strides_in_right_n,
                                                 int64_t         const * i_strides_in_right_k,
                                                 int64_t         const * i_strides_out_aux_c,
                                                 int64_t         const * i_strides_out_aux_m,
                                                 int64_t         const * i_strides_out_aux_n,
                                                 int64_t         const * i_strides_out_c,
                                                 int64_t         const * i_strides_out_m,
                                                 int64_t         const * i_strides_out_n,
                                                 int64_t                 i_num_bytes_scalar_left,
                                                 int64_t                 i_num_bytes_scalar_right,
                                                 int64_t                 i_num_bytes_scalar_out,
                                                 kernel_t                i_ktype_first_touch,
                                                 kernel_t                i_ktype_main,
                                                 kernel_t                i_ktype_last_touch,
                                                 ContractionPackingTpp * i_packing ) {
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

  m_num_bytes_scalar_left  = i_num_bytes_scalar_left;
  m_num_bytes_scalar_right = i_num_bytes_scalar_right;
  m_num_bytes_scalar_out   = i_num_bytes_scalar_out;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main        = i_ktype_main;
  m_ktype_last_touch  = i_ktype_last_touch;

  m_num_tasks_targeted = 1;

  m_packing = i_packing;

  m_threading_first_last_touch = false;

  m_compiled = false;
}

einsum_ir::err_t einsum_ir::backend::ContractionLoops::compile() {
  // determine if the outermost C dimension is complex
  m_cpx_outer_c = false;
  m_cpx_outer_c |= ce_cpx_op( m_ktype_first_touch );
  m_cpx_outer_c |= ce_cpx_op( m_ktype_main );
  m_cpx_outer_c |= ce_cpx_op( m_ktype_last_touch );

  // check if a complex C dimension is possible
  if( m_cpx_outer_c && m_num_dims_c == 0 ) {
    return err_t::INVALID_CPX_DIM;
  }
  if( m_cpx_outer_c && m_sizes_c[0] != 2 ) {
    return err_t::INVALID_CPX_DIM;
  }
  // derive complex strides
  if( m_cpx_outer_c ) {
    m_cpx_stride_in_left_bytes  = m_strides_in_left_c[0]  * m_num_bytes_scalar_left;
    m_cpx_stride_in_right_bytes = m_strides_in_right_c[0] * m_num_bytes_scalar_right;
    m_cpx_stride_out_aux_bytes  = m_strides_out_aux_c[0]  * m_num_bytes_scalar_out;
    m_cpx_stride_out_bytes      = m_strides_out_c[0]      * m_num_bytes_scalar_out;
  }
  else {
    m_cpx_stride_in_left_bytes  = 0;
    m_cpx_stride_in_right_bytes = 0;
    m_cpx_stride_out_aux_bytes  = 0;
    m_cpx_stride_out_bytes      = 0;
  }

  // derive loop parameters for C dimension
  int64_t         l_num_dims_c         = m_cpx_outer_c ? m_num_dims_c         - 1 : m_num_dims_c;
  int64_t const * l_sizes_c            = m_cpx_outer_c ? m_sizes_c            + 1 : m_sizes_c;
  int64_t const * l_strides_in_left_c  = m_cpx_outer_c ? m_strides_in_left_c  + 1 : m_strides_in_left_c;
  int64_t const * l_strides_in_right_c = m_cpx_outer_c ? m_strides_in_right_c + 1 : m_strides_in_right_c;
  int64_t const * l_strides_out_aux_c  = m_cpx_outer_c ? m_strides_out_aux_c  + 1 : m_strides_out_aux_c;
  int64_t const * l_strides_out_c      = m_cpx_outer_c ? m_strides_out_c      + 1 : m_strides_out_c;


  m_num_loops = l_num_dims_c + m_num_dims_m + m_num_dims_n + m_num_dims_k;

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
  for( int64_t l_di = 0; l_di < l_num_dims_c; l_di++ ) {
    if( l_di == l_num_dims_c-1 && m_num_dims_k == 0 && m_num_dims_m == 0 && m_num_dims_n == 0 ) {
      m_loop_first_last_touch.push_back( EVERY_ITER );
    }
    else {
      m_loop_first_last_touch.push_back( NONE );
    }
    m_loop_dim_type.push_back(        dim_t::C                   );
    m_loop_sizes.push_back(           l_sizes_c[l_di]            );
    m_loop_strides_left.push_back(    l_strides_in_left_c[l_di]  );
    m_loop_strides_right.push_back(   l_strides_in_right_c[l_di] );
    m_loop_strides_out_aux.push_back( l_strides_out_aux_c[l_di]  );
    m_loop_strides_out.push_back(     l_strides_out_c[l_di]      );
  }

  // add data of N dimensions
  for( int64_t l_di = 0; l_di < m_num_dims_n; l_di++ ) {
    if( l_di == m_num_dims_n-1 && m_num_dims_k == 0 && m_num_dims_m == 0 ) {
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
    if( l_di == m_num_dims_m-1 && m_num_dims_k == 0 ) {
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
    m_num_loops = 1;
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

  //setup packing loops
  m_id_packing_loop_left = m_num_loops;
  m_id_packing_loop_right = m_num_loops;
  if( m_packing != nullptr ){
    m_id_packing_loop_left -= m_packing->m_packing_loop_offset_left;
    m_id_packing_loop_right -= m_packing->m_packing_loop_offset_right;
  }
  //TODO order loop correctly
  //(use map find for elements and reduce all these loops to one)

  // compile iteration spaces
  err_t l_err = threading( m_num_tasks_targeted );
  if( l_err != err_t::SUCCESS ) {
    return err_t::COMPILATION_FAILED;
  }

  m_compiled = true;

  return err_t::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::ContractionLoops::threading( int64_t i_num_tasks ) {
  m_num_tasks = i_num_tasks;

  int64_t l_num_parallel = m_num_dims_c + m_num_dims_m + m_num_dims_n;
  l_num_parallel = m_cpx_outer_c ? l_num_parallel - 1 : l_num_parallel;

  m_iter_spaces.init( m_num_loops,
                      l_num_parallel,
                      nullptr,
                      m_loop_sizes.data(),
                      m_num_tasks );
  err_t l_err = m_iter_spaces.compile();
  if( l_err != err_t::SUCCESS ) {
    return err_t::COMPILATION_FAILED;
  }

  m_num_tasks = m_iter_spaces.num_tasks();

  return err_t::SUCCESS;
}

void einsum_ir::backend::ContractionLoops::contract_iter( int64_t         i_id_task,
                                                          int64_t         i_id_loop,
                                                          void    const * i_ptr_left,
                                                          void    const * i_ptr_right,
                                                          void    const * i_ptr_out_aux,
                                                          void          * i_ptr_out ) {
  // derive first element and number of iterations
  int64_t l_first = m_iter_spaces.firsts(i_id_task)[ i_id_loop ];
  int64_t l_size  = m_iter_spaces.sizes(i_id_task)[  i_id_loop ];

  // execute first touch kernel
  if( m_loop_first_last_touch[i_id_loop] == BEFORE_AFTER_ITER ) {
    kernel_first_touch( i_ptr_out_aux,
                        i_ptr_out );
  }

  // issue loop iterations
  for( int64_t l_it = l_first; l_it < l_first+l_size; l_it++ ) {
    char * l_ptr_left    = (char *) i_ptr_left;
    char * l_ptr_right   = (char *) i_ptr_right;
    char * l_ptr_out_aux = (char *) i_ptr_out_aux;
    char * l_ptr_out     = (char *) i_ptr_out;

    l_ptr_left    += l_it * m_loop_strides_left[    i_id_loop ];
    l_ptr_right   += l_it * m_loop_strides_right[   i_id_loop ];
    if( l_ptr_out_aux != nullptr ) {
      l_ptr_out_aux += l_it * m_loop_strides_out_aux[ i_id_loop ];
    }
    l_ptr_out     += l_it * m_loop_strides_out[     i_id_loop ];

    if( m_loop_first_last_touch[i_id_loop] == EVERY_ITER ) {
      kernel_first_touch( l_ptr_out_aux,
                          l_ptr_out );
    }

    if( m_id_packing_loop_left == i_id_loop ){
      l_ptr_left = m_packing->kernel_pack_left( l_ptr_left );
    }
    if( m_id_packing_loop_right == i_id_loop ){
      l_ptr_right = m_packing->kernel_pack_right( l_ptr_right );
    }

    if( i_id_loop + 1 < m_num_loops ) {
      contract_iter( i_id_task,
                     i_id_loop+1,
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

void einsum_ir::backend::ContractionLoops::contract( void const * i_tensor_left,
                                                     void const * i_tensor_right,
                                                     void const * i_tensor_out_aux,
                                                     void       * io_tensor_out ) {

if( m_packing != nullptr){
  m_packing->allocate_memory();
}
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for( int64_t l_ta = 0; l_ta < m_num_tasks; l_ta++ ) {
    contract_iter( l_ta,
                   0,
                   i_tensor_left,
                   i_tensor_right,
                   i_tensor_out_aux,
                   io_tensor_out );
  }
}