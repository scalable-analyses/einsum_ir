#include "UnaryLoops.h"

void einsum_ir::backend::UnaryLoops::init( int64_t         i_num_dims,
                                           int64_t const * i_sizes,
                                           int64_t const * i_strides_in,
                                           int64_t const * i_strides_out,
                                           int64_t         i_num_bytes_in,
                                           int64_t         i_num_bytes_out ) {
  m_num_dims             = i_num_dims;
  m_sizes                = i_sizes;
  m_strides_in           = i_strides_in;
  m_strides_out          = i_strides_out;
  m_num_bytes_scalar_in  = i_num_bytes_in;
  m_num_bytes_scalar_out = i_num_bytes_out;
}

einsum_ir::err_t einsum_ir::backend::UnaryLoops::compile() {
  m_compiled = true;
  return err_t::SUCCESS;
}

void einsum_ir::backend::UnaryLoops::threading( int64_t i_num_tasks_target ) {
  if( i_num_tasks_target < 2 ) {
    m_threading_num_loops = -1;
    return;
  }

  int64_t l_num_tasks = 1;

  for( int64_t l_lo = 0; l_lo < m_num_dims; l_lo++ ) {
    l_num_tasks *= m_sizes[l_lo];
    m_threading_num_loops = l_lo+1;

    if( l_num_tasks >= i_num_tasks_target ) {
      break;
    }
  }
}

void einsum_ir::backend::UnaryLoops::eval_iter( int64_t      i_id_loop,
                                                void const * i_ptr_in,
                                                void       * io_ptr_out ) {
  // issue loop iterations
  for( int64_t l_it = 0; l_it < m_sizes[i_id_loop]; l_it++ ) {
    char * l_ptr_in  = (char *) i_ptr_in;
    char * l_ptr_out = (char *) io_ptr_out;

    l_ptr_in  += l_it * m_strides_in[  i_id_loop ] * m_num_bytes_scalar_in;
    l_ptr_out += l_it * m_strides_out[ i_id_loop ] * m_num_bytes_scalar_out;

    if( i_id_loop + 1 < m_num_dims ) {
      eval_iter( i_id_loop+1,
                 l_ptr_in,
                 l_ptr_out );
    }
    else {
      // execute main kernel
      kernel_main( l_ptr_in,
                   l_ptr_out);
    }
  }
}

void einsum_ir::backend::UnaryLoops::eval_iter_parallel_1( int64_t      i_id_loop,
                                                           void const * i_ptr_in,
                                                           void       * io_ptr_out ) {
  // issue loop iterations
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for( int64_t l_it = 0; l_it < m_sizes[i_id_loop]; l_it++ ) {
    char * l_ptr_in  = (char *) i_ptr_in;
    char * l_ptr_out = (char *) io_ptr_out;

    l_ptr_in  += l_it * m_strides_in[  i_id_loop ] * m_num_bytes_scalar_in;
    l_ptr_out += l_it * m_strides_out[ i_id_loop ] * m_num_bytes_scalar_out;

    if( i_id_loop + 1 < m_num_dims ) {
      eval_iter( i_id_loop+1,
                 l_ptr_in,
                 l_ptr_out );
    }
    else {
      // execute main kernel
      kernel_main( l_ptr_in,
                   l_ptr_out);
    }
  }
}

void einsum_ir::backend::UnaryLoops::eval_iter_parallel_2( int64_t      i_id_loop,
                                                           void const * i_ptr_in,
                                                           void       * io_ptr_out ) {
  // issue loop iterations
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for( int64_t l_it_0 = 0; l_it_0 < m_sizes[i_id_loop]; l_it_0++ ) {
    for( int64_t l_it_1 = 0; l_it_1 < m_sizes[i_id_loop+1]; l_it_1++ ) {
      char * l_ptr_in  = (char *) i_ptr_in;
      char * l_ptr_out = (char *) io_ptr_out;

      l_ptr_in  += l_it_0 * m_strides_in[  i_id_loop   ] * m_num_bytes_scalar_in;
      l_ptr_out += l_it_0 * m_strides_out[ i_id_loop   ] * m_num_bytes_scalar_out;

      l_ptr_in  += l_it_1 * m_strides_in[  i_id_loop+1 ] * m_num_bytes_scalar_in;
      l_ptr_out += l_it_1 * m_strides_out[ i_id_loop+1 ] * m_num_bytes_scalar_out;

      if( i_id_loop + 2 < m_num_dims ) {
        eval_iter( i_id_loop+2,
                   l_ptr_in,
                   l_ptr_out );
      }
      else {
        // execute main kernel
        kernel_main( l_ptr_in,
                     l_ptr_out);
      }
    }
  }
}

void einsum_ir::backend::UnaryLoops::eval_iter_parallel_3( int64_t      i_id_loop,
                                                           void const * i_ptr_in,
                                                           void       * io_ptr_out ) {
  // issue loop iterations
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for( int64_t l_it_0 = 0; l_it_0 < m_sizes[i_id_loop]; l_it_0++ ) {
    for( int64_t l_it_1 = 0; l_it_1 < m_sizes[i_id_loop+1]; l_it_1++ ) {
      for( int64_t l_it_2 = 0; l_it_2 < m_sizes[i_id_loop+2]; l_it_2++ ) {
        char * l_ptr_in  = (char *) i_ptr_in;
        char * l_ptr_out = (char *) io_ptr_out;

        l_ptr_in  += l_it_0 * m_strides_in[  i_id_loop   ] * m_num_bytes_scalar_in;
        l_ptr_out += l_it_0 * m_strides_out[ i_id_loop   ] * m_num_bytes_scalar_out;

        l_ptr_in  += l_it_1 * m_strides_in[  i_id_loop+1 ] * m_num_bytes_scalar_in;
        l_ptr_out += l_it_1 * m_strides_out[ i_id_loop+1 ] * m_num_bytes_scalar_out;

        l_ptr_in  += l_it_2 * m_strides_in[  i_id_loop+2 ] * m_num_bytes_scalar_in;
        l_ptr_out += l_it_2 * m_strides_out[ i_id_loop+2 ] * m_num_bytes_scalar_out;

        if( i_id_loop + 3 < m_num_dims ) {
          eval_iter( i_id_loop+3,
                     l_ptr_in,
                     l_ptr_out );
        }
        else {
          // execute main kernel
          kernel_main( l_ptr_in,
                       l_ptr_out);
        }
      }
    }
  }
}

void einsum_ir::backend::UnaryLoops::eval_iter_parallel_4( int64_t      i_id_loop,
                                                           void const * i_ptr_in,
                                                           void       * io_ptr_out ) {
  // issue loop iterations
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
  for( int64_t l_it_0 = 0; l_it_0 < m_sizes[i_id_loop]; l_it_0++ ) {
    for( int64_t l_it_1 = 0; l_it_1 < m_sizes[i_id_loop+1]; l_it_1++ ) {
      for( int64_t l_it_2 = 0; l_it_2 < m_sizes[i_id_loop+2]; l_it_2++ ) {
        for( int64_t l_it_3 = 0; l_it_3 < m_sizes[i_id_loop+3]; l_it_3++ ) {
          char * l_ptr_in  = (char *) i_ptr_in;
          char * l_ptr_out = (char *) io_ptr_out;

          l_ptr_in  += l_it_0 * m_strides_in[  i_id_loop   ] * m_num_bytes_scalar_in;
          l_ptr_out += l_it_0 * m_strides_out[ i_id_loop   ] * m_num_bytes_scalar_out;

          l_ptr_in  += l_it_1 * m_strides_in[  i_id_loop+1 ] * m_num_bytes_scalar_in;
          l_ptr_out += l_it_1 * m_strides_out[ i_id_loop+1 ] * m_num_bytes_scalar_out;

          l_ptr_in  += l_it_2 * m_strides_in[  i_id_loop+2 ] * m_num_bytes_scalar_in;
          l_ptr_out += l_it_2 * m_strides_out[ i_id_loop+2 ] * m_num_bytes_scalar_out;

          l_ptr_in  += l_it_3 * m_strides_in[  i_id_loop+3 ] * m_num_bytes_scalar_in;
          l_ptr_out += l_it_3 * m_strides_out[ i_id_loop+3 ] * m_num_bytes_scalar_out;

          if( i_id_loop + 4 < m_num_dims ) {
            eval_iter( i_id_loop+4,
                       l_ptr_in,
                       l_ptr_out );
          }
          else {
            // execute main kernel
            kernel_main( l_ptr_in,
                         l_ptr_out);
          }
        }
      }
    }
  }
}

void einsum_ir::backend::UnaryLoops::eval( void const * i_tensor_in,
                                           void       * io_tensor_out ) {
  if( m_threading_num_loops < 1 ) {
    eval_iter( 0,
               i_tensor_in,
               io_tensor_out );
  }
  else {
    if( m_threading_num_loops == 1 ) {
      eval_iter_parallel_1( 0,
                            i_tensor_in,
                            io_tensor_out );
    }
    else if( m_threading_num_loops == 2 ) {
      eval_iter_parallel_2( 0,
                            i_tensor_in,
                            io_tensor_out );
    }
    if( m_threading_num_loops == 3 ) {
      eval_iter_parallel_3( 0,
                            i_tensor_in,
                            io_tensor_out );
    }
    else {
      eval_iter_parallel_4( 0,
                            i_tensor_in,
                            io_tensor_out );
    }
  }
}