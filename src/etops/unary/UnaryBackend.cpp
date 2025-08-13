#include "UnaryBackend.h"

void einsum_ir::etops::UnaryBackend::init( std::vector< exec_t >  const & i_exec_type,
                                           std::vector< int64_t > const & i_dim_sizes,
                                           std::vector< int64_t > const & i_strides_in,
                                           std::vector< int64_t > const & i_strides_out,
                                           data_t                         i_dtype_in,
                                           data_t                         i_dtype_comp,
                                           data_t                         i_dtype_out,
                                           kernel_t                       i_ktype,
                                           int64_t                        i_num_threads ){

  //copy to local variables
  m_exec_type = i_exec_type;
  m_dim_sizes = i_dim_sizes;

  m_strides_in  = i_strides_in;
  m_strides_out = i_strides_out;
  
  m_dtype_in   = i_dtype_in;
  m_dtype_out  = i_dtype_out;
  m_dtype_comp = i_dtype_comp;

  m_ktype = i_ktype;

  m_num_threads = i_num_threads;
}

void einsum_ir::etops::UnaryBackend::init( std::vector< iter_property > const & i_iterations,
                                           data_t                               i_dtype_in,
                                           data_t                               i_dtype_comp,
                                           data_t                               i_dtype_out,
                                           kernel_t                             i_ktype,
                                           int64_t                              i_num_threads ){

  int64_t l_num_iters = i_iterations.size();
  m_exec_type.resize(l_num_iters);
  m_dim_sizes.resize(l_num_iters); 
  m_strides_in.resize(l_num_iters);
  m_strides_out.resize(l_num_iters);

  for(int64_t l_id = 0; l_id < l_num_iters; l_id++){
    m_exec_type[   l_id] = i_iterations[l_id].exec_type;
    m_dim_sizes[   l_id] = i_iterations[l_id].size;
    m_strides_in[  l_id] = i_iterations[l_id].stride_left;
    m_strides_out[ l_id] = i_iterations[l_id].stride_out;  
  }

  m_dtype_in   = i_dtype_in;
  m_dtype_out  = i_dtype_out;
  m_dtype_comp = i_dtype_comp;

  m_ktype = i_ktype;

  m_num_threads = i_num_threads;
}

einsum_ir::etops::err_t einsum_ir::etops::UnaryBackend::compile(){
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
  
  //find first parallel loop, number of parallel loops and first primitive loop
  m_id_first_parallel_loop = -1;
  m_id_first_primitive_dim = -1;
  m_num_parallel_loops = 0;
  int64_t l_num_iters = m_exec_type.size();
  for(int64_t l_id = 0; l_id < l_num_iters; l_id++){
    if( m_exec_type.at(l_id) == exec_t::OMP ){
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
  // only parallelization starting with the first loop is supported
  if(m_id_first_parallel_loop > 0){
    return err_t::COMPILATION_FAILED; 
  }

  //multiply strides by size of datatype 
  for(int64_t l_id = 0; l_id < l_num_iters; l_id++){
    m_strides_in[l_id]  *= ce_n_bytes(m_dtype_in );
    m_strides_out[l_id] *= ce_n_bytes(m_dtype_out);

  }

  return err_t::SUCCESS;
}

void einsum_ir::etops::UnaryBackend::contract( void const * i_tensor_in,
                                               void       * io_tensor_out ) {
  if(m_id_first_primitive_dim == 0){
    kernel_main( (char *) i_tensor_in,
                 (char *) io_tensor_out );
  }
  else if(m_id_first_parallel_loop == 0){
    contract_iter_parallel( 0,
                            (char *) i_tensor_in,
                            (char *) io_tensor_out );
  }
  else{
    contract_iter( 0,
                   (char *) i_tensor_in,
                   (char *) io_tensor_out );
  }
}


void einsum_ir::etops::UnaryBackend::contract_iter( int64_t         i_id_loop,
                                                    char    const * i_ptr_in,
                                                    char          * i_ptr_out ) {

  int64_t l_size = m_dim_sizes[i_id_loop];

  // issue loop iterations
  for( int64_t l_it = 0; l_it < l_size; l_it++ ) {
    if( i_id_loop + 1 < m_id_first_primitive_dim ) {
      contract_iter( i_id_loop+1,
                     i_ptr_in,
                     i_ptr_out );
    }
    else {
      // execute main kernel
      kernel_main( i_ptr_in,
                   i_ptr_out );
    }
    i_ptr_in  += m_strides_in[  i_id_loop ];
    i_ptr_out += m_strides_out[ i_id_loop ];
  }
}

void einsum_ir::etops::UnaryBackend::contract_iter_parallel( int64_t         i_id_loop,
                                                             char    const * i_ptr_in,
                                                             char          * i_ptr_out ) {

  int64_t l_all_size = 1;
  for( int64_t l_loop = 0; l_loop < m_num_parallel_loops; l_loop++ ) {
    l_all_size *= m_dim_sizes[l_loop];
  }
  

  // issue loop iterations
#ifdef _OPENMP
#pragma omp parallel for num_threads(m_num_threads)
#endif
  for( int64_t l_it = 0; l_it < l_all_size; l_it++ ) {

    char const * l_ptr_in  = i_ptr_in;
    char       * l_ptr_out = i_ptr_out;

    int64_t l_it_all_loops   = l_it;
    int64_t l_it_single_loop = 0;
    for( int64_t l_loop = m_num_parallel_loops - 1; l_loop >= 0; l_loop-- ) {
      l_it_single_loop = l_it_all_loops % m_dim_sizes[l_loop];
      l_it_all_loops   = l_it_all_loops / m_dim_sizes[l_loop];

      //update pointer
      l_ptr_in  = i_ptr_in  + l_it_single_loop * m_strides_in[  l_loop ];
      l_ptr_out = i_ptr_out + l_it_single_loop * m_strides_out[ l_loop ];
    }

    if( i_id_loop + 1 < m_id_first_primitive_dim ) {
      contract_iter( i_id_loop+1,
                     l_ptr_in,
                     l_ptr_out );
    }
    else {
      // execute main kernel
      kernel_main( l_ptr_in,
                   l_ptr_out );
    }
  }
}

einsum_ir::etops::err_t einsum_ir::etops::UnaryBackend::set_kernel_shape( ){

  //check that there are enough primitive dimensions
  int64_t l_size = m_dim_sizes.size();
  int64_t l_num_prims = 0;
  for( int64_t l_id = l_size - 1; l_id >= 0; l_id-- ){
    if( m_exec_type[l_id] != exec_t::PRIM ){
      break;
    }
    l_num_prims++;
  }
  if( l_num_prims != 2 ){
    return err_t::COMPILATION_FAILED; 
  }

  m_m = m_dim_sizes[l_size-1];
  m_n = m_dim_sizes[l_size-2];

  if( m_strides_in[l_size-1] == 1 && m_strides_out[l_size-1] == 1 ){
    m_lda = m_strides_in[l_size-2];
    m_ldb = m_strides_out[l_size-2];
  }
  else if(m_strides_in[l_size-1] == 1 && m_strides_out[l_size-2] == 1){
    m_lda = m_strides_in[l_size-2];
    m_ldb = m_strides_out[l_size-1];
    m_trans_a = true;
  }
  else{
    return err_t::COMPILATION_FAILED; 
  }

  return err_t::SUCCESS;
}