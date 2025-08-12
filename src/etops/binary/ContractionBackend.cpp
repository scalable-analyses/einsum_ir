#include "ContractionBackend.h"
#include "../unary/UnaryOptimizer.h"
#include <algorithm>

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

  m_is_compiled = false;
}

void einsum_ir::binary::ContractionBackend::init( std::vector< iter_property > const & i_iterations,
                                                  data_t                               i_dtype_left,
                                                  data_t                               i_dtype_right,
                                                  data_t                               i_dtype_comp,
                                                  data_t                               i_dtype_out,
                                                  kernel_t                             i_ktype_first_touch,
                                                  kernel_t                             i_ktype_main,
                                                  kernel_t                             i_ktype_last_touch,
                                                  int64_t                              i_num_threads,
                                                  ContractionMemoryManager           * i_contraction_mem ){

  size_t l_num_iters = i_iterations.size();
  m_dim_type.resize(l_num_iters);
  m_exec_type.resize(l_num_iters);
  m_dim_sizes.resize(l_num_iters); 
  m_strides_left.resize(l_num_iters);
  m_strides_right.resize(l_num_iters);
  m_strides_out.resize(l_num_iters);
  m_strides_out_aux.resize(l_num_iters);
  m_packing_strides_left.resize(l_num_iters);
  m_packing_strides_right.resize(l_num_iters);

  for(size_t l_id = 0; l_id < l_num_iters; l_id++){
    m_dim_type[             l_id] = i_iterations[l_id].dim_type;
    m_exec_type[            l_id] = i_iterations[l_id].exec_type;
    m_dim_sizes[            l_id] = i_iterations[l_id].size;
    m_strides_left[         l_id] = i_iterations[l_id].stride_left;
    m_strides_right[        l_id] = i_iterations[l_id].stride_right;
    m_strides_out[          l_id] = i_iterations[l_id].stride_out;  
    m_strides_out_aux[      l_id] = i_iterations[l_id].stride_out_aux;
    m_packing_strides_left[ l_id] = i_iterations[l_id].packing_stride_left;
    m_packing_strides_right[l_id] = i_iterations[l_id].packing_stride_right;
  }

  m_dtype_left  = i_dtype_left;
  m_dtype_right = i_dtype_right;
  m_dtype_out   = i_dtype_out;
  m_dtype_comp  = i_dtype_comp;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main        = i_ktype_main;
  m_ktype_last_touch  = i_ktype_last_touch;

  m_num_threads = i_num_threads;
  m_memory = i_contraction_mem;

  m_is_compiled = false;
}

einsum_ir::err_t einsum_ir::binary::ContractionBackend::compile(){
  err_t l_err = err_t::UNDEFINED_ERROR;
  if( m_is_compiled ){
    return err_t::SUCCESS;
  }

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
  if( m_num_parallel_loops == 0 ){
    m_num_threads = 1;
  }

  //check if first and last touch exists
  m_has_first_touch = m_ktype_first_touch != kernel_t::UNDEFINED_KTYPE;
  m_has_last_touch = m_ktype_last_touch != kernel_t::UNDEFINED_KTYPE;

  //create packing
  create_packing( m_packing_left_id,
                  m_size_packing_left,
                  m_unary_left,
                  m_strides_left,
                  m_packing_strides_left);
  m_size_packing_left *= ce_n_bytes(m_dtype_left);
  
  create_packing( m_packing_right_id,
                  m_size_packing_right,
                  m_unary_right,
                  m_strides_right,
                  m_packing_strides_right);
  m_size_packing_right *= ce_n_bytes(m_dtype_right);

  //reserve memory for packing
  int64_t l_reserved_size = m_size_packing_left * m_num_cached_ptrs_left + m_size_packing_right * m_num_cached_ptrs_right;
  if( m_memory != nullptr ){
    m_memory->reserve_thread_memory( l_reserved_size, m_num_threads );
  }else if( l_reserved_size > 0 ){
    return err_t::COMPILATION_FAILED;
  }

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
               m_num_threads );

  // compile iteration spaces
  l_err = m_iter.setup( m_strides_left,
                        m_strides_right,
                        m_strides_out_aux,
                        m_strides_out,
                        m_thread_infos );

  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  m_is_compiled = true;
  return err_t::SUCCESS;
}

void einsum_ir::binary::ContractionBackend::contract( void const * i_tensor_left,
                                                      void const * i_tensor_right,
                                                      void const * i_tensor_out_aux,
                                                      void       * io_tensor_out ) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(m_num_threads)
#endif
  for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ) {
    thread_info * l_thread_inf = &m_thread_infos[l_thread_id];
    //get packing memory
    if( m_size_packing_left || m_size_packing_right ){
      l_thread_inf->memory_left  = m_memory->get_thread_memory( l_thread_id );
      l_thread_inf->memory_right = l_thread_inf->memory_left + m_size_packing_left * m_num_cached_ptrs_left;
      l_thread_inf->cached_ptrs_left.resize(  m_num_cached_ptrs_left,  nullptr );
      l_thread_inf->cached_ptrs_right.resize( m_num_cached_ptrs_right, nullptr );
    }

    //contract
    contract_iter( l_thread_inf,
                   0,
                   (char *) i_tensor_left    + l_thread_inf->offset_left,
                   (char *) i_tensor_right   + l_thread_inf->offset_right,
                   (char *) i_tensor_out_aux + l_thread_inf->offset_out_aux,
                   (char *) io_tensor_out    + l_thread_inf->offset_out,
                   m_has_first_touch,
                   m_has_last_touch );
  }
}

void einsum_ir::binary::ContractionBackend::contract_iter( thread_info   * i_thread_inf,
                                                           int64_t         i_id_loop,
                                                           char    const * i_ptr_left,
                                                           char    const * i_ptr_right,
                                                           char    const * i_ptr_out_aux,
                                                           char          * i_ptr_out,
                                                           bool            i_first_access,
                                                           bool            i_last_access ) {
  bool l_first_access = i_first_access;
  bool l_last_access  = i_last_access;

  int64_t l_current_id = i_id_loop;
  int64_t l_direction = 1;

  int64_t l_size = m_dim_sizes[i_id_loop];
  int64_t l_id_next_loop = i_id_loop + 1;
  if( i_id_loop == m_id_first_parallel_loop ){
    l_id_next_loop = i_id_loop + m_num_parallel_loops;
    l_size = i_thread_inf->movement_ids.size();
  }

  uint64_t l_id_m = 0x7FFFFFFFFFFFFFFF;
  uint64_t l_id_n = 0x7FFFFFFFFFFFFFFF;

  // issue loop iterations
  for( int64_t l_it = 0; l_it < l_size; l_it++ ) {
    bool l_non_k_loop = m_dim_type[i_id_loop] != dim_t::K;
    l_first_access = i_first_access && ( l_non_k_loop || l_it == 0 );
    l_last_access  = i_last_access  && ( l_non_k_loop || l_it == m_dim_sizes[i_id_loop] - 1 );

    //calculate sfc current id and direction
    if( i_id_loop == m_id_first_parallel_loop ){
      sfc_t l_move =  i_thread_inf->movement_ids[l_it];
      sfc_t l_sign = (l_move & 1);
      l_direction  = 1 - ( (int64_t)l_sign << 1); 
      l_current_id = l_move >> 1;
    }

    //pack left tensor
    const char * l_ptr_left_active = i_ptr_left;
    if( m_packing_left_id == l_id_next_loop )  {
      int64_t l_id = l_id_m % m_num_cached_ptrs_left;
      l_ptr_left_active = i_thread_inf->memory_left + l_id * m_size_packing_left;
      if( i_ptr_left != i_thread_inf->cached_ptrs_left[l_id] ){
        m_unary_left.contract(i_ptr_left, (void *)l_ptr_left_active);
        i_thread_inf->cached_ptrs_left[l_id] = i_ptr_left;
      }
    }

    //pack right tensor
    const char * l_ptr_right_active = i_ptr_right;
    if( m_packing_right_id == l_id_next_loop )  {
      int64_t l_id = l_id_n % m_num_cached_ptrs_right;
      l_ptr_right_active = i_thread_inf->memory_right + l_id * m_size_packing_left;
      if( i_ptr_right != i_thread_inf->cached_ptrs_right[l_id]){
        m_unary_right.contract(i_ptr_right, (void *)l_ptr_right_active);
        i_thread_inf->cached_ptrs_right[l_id] = i_ptr_right;
      }
    }

    if( l_id_next_loop < m_id_first_primitive_dim ) {
      contract_iter( i_thread_inf,
                     l_id_next_loop,
                     l_ptr_left_active,
                     l_ptr_right_active,
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
      kernel_main( l_ptr_left_active,
                   l_ptr_right_active,
                   i_ptr_out );
      
      if( l_last_access ) {
        kernel_last_touch( i_ptr_out_aux,
                           i_ptr_out );
      }
    }

    //update pointer
    i_ptr_left    += l_direction * m_strides_left[    l_current_id ];
    i_ptr_right   += l_direction * m_strides_right[   l_current_id ];
    i_ptr_out_aux += l_direction * m_strides_out_aux[ l_current_id];
    i_ptr_out     += l_direction * m_strides_out[     l_current_id ];

    bool l_is_m_loop = m_dim_type[l_current_id] == dim_t::M;
    bool l_is_n_loop = m_dim_type[l_current_id] == dim_t::N;

    l_id_m += l_direction * l_is_m_loop;
    l_id_n += l_direction * l_is_n_loop;
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
  if( m_ktype_main == kernel_t::PACKED_MADD ||
      m_ktype_main == kernel_t::CPX_PACKED_MADD ){
    m_r = m_dim_sizes[l_id_packed];
    m_packed_stride_a = m_strides_left[l_id_packed];
    m_packed_stride_b = m_strides_right[l_id_packed];
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

einsum_ir::err_t einsum_ir::binary::ContractionBackend::create_packing( int64_t              & o_packing_id,
                                                                        int64_t              & o_size_packing,
                                                                        UnaryBackendTpp      & o_unary,
                                                                        std::vector<int64_t> & i_strides,
                                                                        std::vector<int64_t> & i_packing_strides ){
  //determine size of and iteration id of packing
  o_packing_id = -1;
  for( std::size_t l_id = 0; l_id < i_packing_strides.size(); l_id++ ) {
    size_t l_reversed_id = i_packing_strides.size() - 1 - l_id;
    if( i_packing_strides[l_reversed_id] != 0 ){
      o_packing_id = l_reversed_id;
      o_size_packing += i_strides[l_reversed_id]  * (m_dim_sizes[l_reversed_id] - 1);
    }
    //if packing is unaffected by outer loop move it to outer loop 
    if(    i_strides[l_reversed_id] == 0
        && (int64_t)l_reversed_id + 1 == o_packing_id 
        && m_exec_type[l_reversed_id] != exec_t::SFC ){
      o_packing_id = l_reversed_id;
    }
  }
  if( o_packing_id >= 0 ){
    o_size_packing  = (o_size_packing  + 1); 
  }

  //create packing kernel
  if( o_packing_id >= 0 ){
    //setup data structure for packing
    std::vector< iter_property > l_packing_iters;
    l_packing_iters.reserve( i_packing_strides.size() );
    for( std::size_t l_id = 0; l_id < i_packing_strides.size(); l_id++ ) {
      if(i_packing_strides[l_id] != 0){
        iter_property iter_prop;
        iter_prop.exec_type     = exec_t::SEQ;
        iter_prop.size          = m_dim_sizes[l_id];
        iter_prop.stride_left   = i_packing_strides[l_id];
        iter_prop.stride_out    = i_strides[l_id];
        l_packing_iters.push_back( iter_prop );
      }
    }
    //optimize packing iters
    UnaryOptimizer l_unary_opt;
    l_unary_opt.init( &l_packing_iters, 1 );
    err_t l_err = err_t::UNDEFINED_ERROR;
    l_err = l_unary_opt.optimize();
    if( l_err != err_t::SUCCESS ) {
      return l_err;
    }

    //init and compile kernel
    o_unary.init(l_packing_iters, m_dtype_left, m_dtype_comp, m_dtype_out, kernel_t::COPY, 1);
    l_err = o_unary.compile();
    if( l_err != err_t::SUCCESS ) {
      return l_err;
    }
  }
  return err_t::SUCCESS;
}