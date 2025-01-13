#include "ContractionBackend.h"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::backend::ContractionBackend::init( std::vector< dim_t >   const & i_loop_dim_type,
                                                   std::vector< exec_t >  const & i_loop_exec_type,
                                                   std::vector< int64_t > const & i_loop_sizes,
                                                   std::vector< int64_t > const & i_loop_strides_left,
                                                   std::vector< int64_t > const & i_loop_strides_right,
                                                   std::vector< int64_t > const & i_loop_strides_out_aux,
                                                   std::vector< int64_t > const & i_loop_strides_out,
                                                   data_t                               i_dtype_left,
                                                   data_t                               i_dtype_right,
                                                   data_t                               i_dtype_comp,
                                                   data_t                               i_dtype_out,
                                                   kernel_t                             i_ktype_first_touch,
                                                   kernel_t                             i_ktype_main,
                                                   kernel_t                             i_ktype_last_touch ){

  m_loop_dim_type        = i_loop_dim_type;
  m_loop_exec_type       = i_loop_exec_type;
  m_loop_sizes           = i_loop_sizes;
  m_loop_strides_left    = i_loop_strides_left;
  m_loop_strides_right   = i_loop_strides_right;
  m_loop_strides_out_aux = i_loop_strides_out_aux;
  m_loop_strides_out     = i_loop_strides_out;

  m_dtype_left  = i_dtype_left;
  m_dtype_right = i_dtype_right;
  m_dtype_comp  = i_dtype_comp;
  m_dtype_out   = i_dtype_out;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main        = i_ktype_main;
  m_ktype_last_touch  = i_ktype_last_touch;
}

einsum_ir::err_t einsum_ir::backend::ContractionBackend::compile(){
  err_t l_err = err_t::UNDEFINED_ERROR;

  // compile kernel
  l_err = compile_kernels();
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }
  
  //find first parallel loop, number of parallel loops and first primitive loop
  m_id_first_parallel_loop = -1;
  m_id_first_primitive_loop = -1;
  m_num_parallel_loops = 0;
  int64_t l_num_loops = m_loop_dim_type.size();
  for(int64_t l_id = 0; l_id < l_num_loops; l_id++){
    if( m_loop_exec_type.at(l_id) == einsum_ir::OMP ||
        m_loop_exec_type.at(l_id) == einsum_ir::SFC    ){
      if( m_id_first_parallel_loop == -1 ){
        m_id_first_parallel_loop = l_id;
      }
      m_num_parallel_loops++;
    }
    if( m_loop_exec_type.at(l_id) == einsum_ir::PRIM ){
      m_id_first_primitive_loop = l_id;
      break;
    }
  }

  //check if first and last touch exists
  if( m_ktype_first_touch != einsum_ir::UNDEFINED_KTYPE ){
    m_has_first_touch = true;
  }
  if( m_ktype_last_touch != einsum_ir::UNDEFINED_KTYPE ){
    m_has_last_touch = true;
  }

  //multiply strides by size of datatype
  int64_t l_num_bytes_scalar_left  = ce_n_bytes(m_dtype_left);
  int64_t l_num_bytes_scalar_right = ce_n_bytes(m_dtype_right);
  int64_t l_num_bytes_scalar_out   = ce_n_bytes(m_dtype_out);
  for(int64_t l_id = 0; l_id < l_num_loops; l_id++){
    m_loop_strides_left[ l_id] *= l_num_bytes_scalar_left;
    m_loop_strides_right[l_id] *= l_num_bytes_scalar_right;
    m_loop_strides_out[  l_id] *= l_num_bytes_scalar_out;
  }

  // init iteration spaces
  int64_t l_num_threads = 1;
#ifdef _OPENMP
  l_num_threads = omp_get_max_threads(); 
#endif

  m_iter.init( &m_loop_dim_type,
               &m_loop_exec_type,
               &m_loop_sizes,
               &m_loop_strides_left,
               &m_loop_strides_right,
               &m_loop_strides_out,
               l_num_threads );

  // compile iteration spaces
  l_err = m_iter.compile();
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  return err_t::SUCCESS;
}

void einsum_ir::backend::ContractionBackend::contract( void const * i_tensor_left,
                                                       void const * i_tensor_right,
                                                       void const * i_tensor_out_aux,
                                                       void       * io_tensor_out ) {
  //only execute in parallel if there are parallel loops
  if( m_id_first_parallel_loop >= 0 ){
#ifdef _OPENMP
#pragma omp parallel
    {
      int64_t l_thread_id = omp_get_thread_num();
      contract_iter( l_thread_id,
                     0,
                     i_tensor_left,
                     i_tensor_right,
                     i_tensor_out_aux,
                     io_tensor_out,
                     m_has_first_touch,
                     m_has_last_touch );
    }
#endif
  }
  else{
    contract_iter( 0,
                   0,
                   i_tensor_left,
                   i_tensor_right,
                   i_tensor_out_aux,
                   io_tensor_out,
                   m_has_first_touch,
                   m_has_last_touch );
  }
}


void einsum_ir::backend::ContractionBackend::contract_iter( int64_t         i_thread_id,
                                                            int64_t         i_id_loop,
                                                            void    const * i_ptr_left,
                                                            void    const * i_ptr_right,
                                                            void    const * i_ptr_out_aux,
                                                            void          * i_ptr_out,
                                                            bool            i_first_access,
                                                            bool            i_last_access ) {

  bool l_first_access = i_first_access;
  bool l_last_access  = i_last_access;

  //TODO remove
  char * l_ptr_left    = (char *) i_ptr_left;
  char * l_ptr_right   = (char *) i_ptr_right;
  char * l_ptr_out_aux = (char *) i_ptr_out_aux;
  char * l_ptr_out     = (char *) i_ptr_out;

  int64_t l_size  = m_loop_sizes[i_id_loop];
  if( i_id_loop == m_id_first_parallel_loop ){
    m_iter.addInitialOffsets( i_thread_id, &l_ptr_left, &l_ptr_right, &l_ptr_out );
    i_id_loop += m_num_parallel_loops - 1;
    l_size = m_iter.getNumTasks( i_thread_id );
  }

  // issue loop iterations
  for( int64_t l_it = 0; l_it < l_size; l_it++ ) {
    if(m_loop_dim_type[i_id_loop] == einsum_ir::K){
      l_first_access = i_first_access && l_it == 0 ;
      l_last_access  = i_last_access  && l_it == m_loop_sizes[i_id_loop]-1 ;
    }

    if( i_id_loop + 1 < m_id_first_primitive_loop ) {
      contract_iter( i_thread_id,
                     i_id_loop+1,
                     l_ptr_left,
                     l_ptr_right,
                     l_ptr_out_aux,
                     l_ptr_out,
                     l_first_access,
                     l_last_access );
    }
    else {
      if( l_first_access ) {
        kernel_first_touch( l_ptr_out_aux,
                            l_ptr_out );
      }
      // execute main kernel
      kernel_main( l_ptr_left,
                   l_ptr_right,
                   l_ptr_out );
      
      if( l_last_access ) {
        kernel_last_touch( l_ptr_out_aux,
                           l_ptr_out );
      }
    }

    //update pointer
    if( i_id_loop == m_id_first_parallel_loop + m_num_parallel_loops - 1 ){
      m_iter.addMovementOffsets(i_thread_id, l_it, &l_ptr_left, &l_ptr_right, &l_ptr_out );
    }
    else{
      l_ptr_left    += m_loop_strides_left[    i_id_loop ];
      l_ptr_right   += m_loop_strides_right[   i_id_loop ];
      l_ptr_out_aux += m_loop_strides_out_aux[ i_id_loop ];
      l_ptr_out     += m_loop_strides_out[     i_id_loop ];
    }  
  }
}
