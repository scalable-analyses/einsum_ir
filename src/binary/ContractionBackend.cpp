#include "ContractionBackend.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::binary::ContractionBackend::init( std::vector< dim_t >   const & i_dim_type,
                                                  std::vector< exec_t >  const & i_exec_type,
                                                  std::vector< int64_t > const & i_dim_sizes,
                                                  std::vector< int64_t > const & i_strides_left,
                                                  std::vector< int64_t > const & i_strides_right,
                                                  std::vector< int64_t > const & i_strides_out_aux,
                                                  std::vector< int64_t > const & i_strides_out,
                                                  data_t                         i_dtype_left,
                                                  data_t                         i_dtype_right,
                                                  data_t                         i_dtype_comp,
                                                  data_t                         i_dtype_out,
                                                  kernel_t                       i_ktype_first_touch,
                                                  kernel_t                       i_ktype_main,
                                                  kernel_t                       i_ktype_last_touch,
                                                  int64_t                        i_num_threads ){

  //copy to local variables
  m_dim_type        = i_dim_type;
  m_exec_type       = i_exec_type;
  m_dim_sizes       = i_dim_sizes;
  m_strides_left    = i_strides_left;
  m_strides_right   = i_strides_right;
  m_strides_out     = i_strides_out;
  m_strides_out_aux = i_strides_out_aux;
  
  m_dtype_left  = i_dtype_left;
  m_dtype_right = i_dtype_right;
  m_dtype_out   = i_dtype_out;
  m_dtype_comp  = i_dtype_comp;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main        = i_ktype_main;
  m_ktype_last_touch  = i_ktype_last_touch;

  m_num_threads = i_num_threads;
}

void einsum_ir::binary::ContractionBackend::init( std::vector< iter_property > const & i_iterations,
                                                  data_t                               i_dtype_left,
                                                  data_t                               i_dtype_right,
                                                  data_t                               i_dtype_comp,
                                                  data_t                               i_dtype_out,
                                                  kernel_t                             i_ktype_first_touch,
                                                  kernel_t                             i_ktype_main,
                                                  kernel_t                             i_ktype_last_touch,
                                                  int64_t                              i_num_threads ){

  int64_t l_num_iters = i_iterations.size();
  m_dim_type.resize(l_num_iters);
  m_exec_type.resize(l_num_iters);
  m_dim_sizes.resize(l_num_iters); 
  m_strides_left.resize(l_num_iters);
  m_strides_right.resize(l_num_iters);
  m_strides_out.resize(l_num_iters);
  m_strides_out_aux.resize(l_num_iters);

  for(int64_t l_id = 0; l_id < l_num_iters; l_id++){
    m_dim_type[       l_id] = i_iterations[l_id].dim_type;
    m_exec_type[      l_id] = i_iterations[l_id].exec_type;
    m_dim_sizes[      l_id] = i_iterations[l_id].size;
    m_strides_left[   l_id] = i_iterations[l_id].stride_left;
    m_strides_right[  l_id] = i_iterations[l_id].stride_right;
    m_strides_out[    l_id] = i_iterations[l_id].stride_out;  
    m_strides_out_aux[l_id] = i_iterations[l_id].stride_out_aux;
  }

  m_dtype_left  = i_dtype_left;
  m_dtype_right = i_dtype_right;
  m_dtype_out   = i_dtype_out;
  m_dtype_comp  = i_dtype_comp;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main        = i_ktype_main;
  m_ktype_last_touch  = i_ktype_last_touch;

  m_num_threads = i_num_threads;
}

einsum_ir::err_t einsum_ir::binary::ContractionBackend::compile(){
  err_t l_err = err_t::UNDEFINED_ERROR;

  // get kernel shape
  l_err = set_kernel_shape();
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  // compile kernel
  l_err = compile_kernels();
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  //create at least one non Primitive loop
  if( m_exec_type.at(0) == exec_t::PRIM ){
    m_dim_type.insert( m_dim_type.begin(),  dim_t::UNDEFINED_DIM);
    m_exec_type.insert(m_exec_type.begin(), exec_t::SEQ);
    m_dim_sizes.insert(m_dim_sizes.begin(), 1);
    m_strides_left.insert(   m_strides_left.begin(),    0); 
    m_strides_right.insert(  m_strides_right.begin(),   0); 
    m_strides_out.insert(    m_strides_out.begin(),     0); 
    m_strides_out_aux.insert(m_strides_out_aux.begin(), 0); 
  }
  
  //find first parallel loop, number of parallel loops and first primitive loop
  m_id_first_parallel_loop = -1;
  m_id_first_primitive_dim = -1;
  m_num_parallel_loops = 0;
  int64_t l_num_iters = m_dim_type.size();
  for(int64_t l_id = 0; l_id < l_num_iters; l_id++){
    if( m_exec_type.at(l_id) == exec_t::OMP ||
        m_exec_type.at(l_id) == exec_t::SFC    ){
      if( m_id_first_parallel_loop == -1 ){
        m_id_first_parallel_loop = l_id;
      }
      m_num_parallel_loops++;
    }
    if( m_exec_type.at(l_id) == exec_t::PRIM ){
      m_id_first_primitive_dim = l_id;
      break;
    }
  }

  //check if first and last touch exists
  m_has_first_touch = m_ktype_first_touch != kernel_t::UNDEFINED_KTYPE;
  m_has_last_touch = m_ktype_last_touch != kernel_t::UNDEFINED_KTYPE;

  //multiply strides by size of datatype 
  for(int64_t l_id = 0; l_id < l_num_iters; l_id++){
    m_strides_left[l_id]    *= ce_n_bytes(m_dtype_left );
    m_strides_right[l_id]   *= ce_n_bytes(m_dtype_right);
    m_strides_out[l_id]     *= ce_n_bytes(m_dtype_out  );
    m_strides_out_aux[l_id] *= ce_n_bytes(m_dtype_out  );
  }
  

  // init iteration spaces
  m_iter.init( &m_dim_type,
               &m_exec_type,
               &m_dim_sizes,
               &m_strides_left,
               &m_strides_right,
               &m_strides_out_aux,
               &m_strides_out,
               m_num_threads );

  // compile iteration spaces
  l_err = m_iter.compile();
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  return err_t::SUCCESS;
}

void einsum_ir::binary::ContractionBackend::contract( void const * i_tensor_left,
                                                      void const * i_tensor_right,
                                                      void const * i_tensor_out_aux,
                                                      void       * io_tensor_out ) {
  //only execute in parallel if there are parallel loops
  if( m_id_first_parallel_loop >= 0 && m_num_threads > 1 ){
#ifdef _OPENMP
#pragma omp parallel num_threads(m_num_threads)
    {
      int64_t l_thread_id = omp_get_thread_num();
      int64_t l_offset_left, l_offset_right, l_offset_out_aux, l_offset_out;
      m_iter.get_initial_offsets( l_thread_id, l_offset_left, l_offset_right, l_offset_out_aux, l_offset_out );
      contract_iter( l_thread_id,
                     0,
                     (char *) i_tensor_left + l_offset_left,
                     (char *) i_tensor_right + l_offset_right,
                     (char *) i_tensor_out_aux + l_offset_out_aux,
                     (char *) io_tensor_out + l_offset_out,
                     m_has_first_touch,
                     m_has_last_touch );
    }
#endif
  }
  else{
    contract_iter( 0,
                   0,
                   (char *) i_tensor_left,
                   (char *) i_tensor_right,
                   (char *) i_tensor_out_aux,
                   (char *) io_tensor_out,
                   m_has_first_touch,
                   m_has_last_touch );
  }
}


void einsum_ir::binary::ContractionBackend::contract_iter( int64_t         i_thread_id,
                                                           int64_t         i_id_loop,
                                                           char    const * i_ptr_left,
                                                           char    const * i_ptr_right,
                                                           char    const * i_ptr_out_aux,
                                                           char          * i_ptr_out,
                                                           bool            i_first_access,
                                                           bool            i_last_access ) {
  bool l_first_access = i_first_access;
  bool l_last_access  = i_last_access;

  int64_t l_size = m_dim_sizes[i_id_loop];
  if( i_id_loop == m_id_first_parallel_loop ){
    i_id_loop += m_num_parallel_loops - 1;
    l_size = m_iter.get_num_tasks( i_thread_id );
  }

  // issue loop iterations
  for( int64_t l_it = 0; l_it < l_size; l_it++ ) {
    bool l_non_k_loop = m_dim_type[i_id_loop] != dim_t::K;
    l_first_access = i_first_access && ( l_non_k_loop || l_it == 0 );
    l_last_access  = i_last_access  && ( l_non_k_loop || l_it == m_dim_sizes[i_id_loop] - 1 );

    if( i_id_loop + 1 < m_id_first_primitive_dim ) {
      contract_iter( i_thread_id,
                     i_id_loop+1,
                     i_ptr_left,
                     i_ptr_right,
                     i_ptr_out_aux,
                     i_ptr_out,
                     l_first_access,
                     l_last_access );
    }
    else {
      if( l_first_access ) {
        kernel_first_touch( i_ptr_out_aux,
                            i_ptr_out );
      }
      // execute main kernel
      kernel_main( i_ptr_left,
                   i_ptr_right,
                   i_ptr_out );
      
      if( l_last_access ) {
        kernel_last_touch( i_ptr_out_aux,
                           i_ptr_out );
      }
    }

    //update pointer
    if( i_id_loop == m_id_first_parallel_loop + m_num_parallel_loops - 1 ){
      m_iter.add_movement_offsets(i_thread_id, l_it, &i_ptr_left, &i_ptr_right, &i_ptr_out_aux, &i_ptr_out );
    }
    else{
      i_ptr_left    += m_strides_left[ i_id_loop ];
      i_ptr_right   += m_strides_right[ i_id_loop ];
      i_ptr_out_aux += m_strides_out_aux[ i_id_loop ];
      i_ptr_out     += m_strides_out[ i_id_loop ];
    }  
  }
}

einsum_ir::err_t einsum_ir::binary::ContractionBackend::set_kernel_shape( ){
  //check that there are enough primitive dimensions
  int64_t l_size = m_dim_sizes.size();
  int64_t l_num_prims = 0;
  for( int64_t l_id = l_size - 1; l_id >= 0; l_id-- ){
    if( m_exec_type[l_id] != exec_t::PRIM ){
      break;
    }
    l_num_prims++;
  }

  if(    ( m_ktype_main == kernel_t::MADD            && l_num_prims != 3 )
      || ( m_ktype_main == kernel_t::BR_MADD         && l_num_prims != 4 )
      || ( m_ktype_main == kernel_t::CPX_MADD        && l_num_prims != 4 )
      || ( m_ktype_main == kernel_t::PACKED_MADD     && l_num_prims != 4 )
      || ( m_ktype_main == kernel_t::CPX_PACKED_MADD && l_num_prims != 5 ) ){
    return err_t::COMPILATION_FAILED; 
  }

  //check that primitive dimensions have the correct type
  if( m_dim_type[l_size-3] != dim_t::M ||
      m_dim_type[l_size-2] != dim_t::N ||
      m_dim_type[l_size-1] != dim_t::K ){
    return err_t::COMPILATION_FAILED; 
  }
  if(    m_ktype_main == kernel_t::BR_MADD 
      && m_dim_type[l_size-4] != dim_t::K ){
      return err_t::COMPILATION_FAILED; 
  }
  if(    m_ktype_main == kernel_t::PACKED_MADD 
      && m_dim_type[l_size-4] != dim_t::C ){
      return err_t::COMPILATION_FAILED; 
  }
  if(    m_ktype_main == kernel_t::CPX_MADD 
      && m_dim_type[l_size-4] != dim_t::CPX ){
      return err_t::COMPILATION_FAILED; 
  }
  if(    m_ktype_main == kernel_t::CPX_PACKED_MADD 
      && m_dim_type[l_size-4] != dim_t::C 
      && m_dim_type[l_size-5] != dim_t::CPX ){
      return err_t::COMPILATION_FAILED; 
  }

  const int64_t l_id_br     = l_size-4;
  const int64_t l_id_packed = l_size-4;
  const int64_t l_id_m      = l_size-3;
  const int64_t l_id_n      = l_size-2;
  const int64_t l_id_k      = l_size-1;

  //set br parameter
  m_br = 1;
  m_br_stride_a = 0;
  m_br_stride_b = 0;
  if( m_ktype_main == kernel_t::BR_MADD ){
    m_br = m_dim_sizes[l_id_br];
    m_br_stride_a = m_strides_left[l_id_br];
    m_br_stride_b = m_strides_right[l_id_br];
  }

  //set packed parameter
  m_r = 1;
  if( m_ktype_main == kernel_t::PACKED_MADD ){
    m_r = m_dim_sizes[l_id_packed];
  }

  //set m, n, k
  m_m  = m_dim_sizes[l_id_m];
  m_n  = m_dim_sizes[l_id_n];
  m_k  = m_dim_sizes[l_id_k];

  //set lda
  if(    m_m == 1 
      || m_strides_left[l_id_m] == (int64_t)m_r
      || m_strides_left[l_id_m] == 1            ){
    m_trans_a = false;
    m_lda = m_strides_left[l_id_k];
  }
  else if(    m_k == 1
           || m_strides_left[l_id_k] == 1 ){
    m_trans_a = true;
    m_lda = m_strides_left[l_id_m];
  }
  else{
    return err_t::COMPILATION_FAILED;
  }

  //set ldb
  if(    m_k == 1
      || m_strides_right[l_id_k] == (int64_t)m_r
      || m_strides_right[l_id_k] == 1            ){
    m_trans_b = false;
    m_ldb = m_strides_right[l_id_n];
  }
  else if(    m_n == 1
           || m_strides_right[l_id_n] == 1 ){
    m_trans_b = true;
    m_ldb = m_strides_right[l_id_k];
  } 
  else{
    return err_t::COMPILATION_FAILED;
  }

  //set ldc
  if(    m_m == 1
      || m_strides_out[l_id_m] == (int64_t)m_r ){
    m_ldc = m_strides_out[l_id_n];
  }
  else{
    return err_t::COMPILATION_FAILED;
  }

  //set auxiliary strides
  if(    m_m == 1
      || m_strides_out_aux[l_id_m] <= (int64_t)m_r ){
    m_stride_m_out_aux = m_strides_out_aux[l_id_m];
    m_stride_n_out_aux = m_strides_out_aux[l_id_n];
  }
  else{
    return err_t::COMPILATION_FAILED;
  }

  //set leading dimensions of size 1 loops for correct kernel generation
  if( m_k == 1 && !m_trans_a ){
    m_lda = m_m * m_r;
  }
  if( m_m == 1 && m_trans_a ){
    m_lda = m_k * m_r;
  }
  if( m_n == 1 && !m_trans_b){
    m_ldb = m_k * m_r;
  }
  if( m_k == 1 && m_trans_b ){
    m_ldb = m_n * m_r;
  }
  if( m_n == 1 ){
    m_ldc = m_m * m_r;
    m_stride_n_out_aux = m_m * m_r;
  }
  if( m_m == 1 ){
    m_stride_m_out_aux = m_r;
  }

  //set complex parameter
  int64_t l_id_cpx = -1;
  l_id_cpx = (m_ktype_main == kernel_t::CPX_MADD       ) ? l_size - 4 : l_id_cpx;
  l_id_cpx = (m_ktype_main == kernel_t::CPX_PACKED_MADD) ? l_size - 5 : l_id_cpx;
  if( l_id_cpx >= 0){
    if( m_dim_sizes[l_id_cpx] != 2 ){
      return err_t::COMPILATION_FAILED; 
    }
    m_cpx_stride_in_left_bytes  = m_strides_left[   l_id_cpx] * ce_n_bytes(m_dtype_left );
    m_cpx_stride_in_right_bytes = m_strides_right[  l_id_cpx] * ce_n_bytes(m_dtype_right);
    m_cpx_stride_out_aux_bytes  = m_strides_out_aux[l_id_cpx] * ce_n_bytes(m_dtype_out  );
    m_cpx_stride_out_bytes      = m_strides_out[    l_id_cpx] * ce_n_bytes(m_dtype_out  );
  }

  return err_t::SUCCESS;
}