#include "ContractionOptimizer.h"
#include <algorithm>
#include <cmath>

#include <iostream>

void einsum_ir::binary::ContractionOptimizer::init( std::vector< iter_property > * i_iter_space,
                                                    kernel_t                     * i_ktype_main,
                                                    int64_t                        i_num_threads,
                                                    int64_t                        i_target_m,
                                                    int64_t                        i_target_n,
                                                    int64_t                        i_target_k,
                                                    bool                           i_br_gemm_support,
                                                    bool                           i_packed_gemm_support,
                                                    int64_t                        i_num_bytes_scalar_out,
                                                    int64_t                        i_l2_cache_size ){
  m_iter_space = i_iter_space;
  m_ktype_main = i_ktype_main;
  m_num_threads = i_num_threads;

  m_target_m = i_target_m;
  m_target_n = i_target_n;
  m_target_k = i_target_k;

  m_br_gemm_support = i_br_gemm_support;
  m_packed_gemm_support = i_packed_gemm_support;

  m_target_parallel = i_num_threads;

  m_num_bytes_scalar_out = i_num_bytes_scalar_out;
  m_l2_cache_size = i_l2_cache_size;
}

void einsum_ir::binary::ContractionOptimizer::optimize(){
  // removes size 1 iters
  remove_empty_iters();

  // sort iters and fuse them afterwards
  sort_and_fuse_iters();

  // find and add the Kernel
  set_primitive_iters();

  // set primitive iters might create new size 1 iters
  remove_empty_iters();

  // reoders, adds and parallelizes the remaining iteration space
  reorder_and_parallelize_iters();
}

void einsum_ir::binary::ContractionOptimizer::sort_and_fuse_iters(){
  //sort iters depending on stride and dimension type
  std::stable_sort( m_iter_space->begin(), m_iter_space->end(), 
                    [&](iter_property l_a, iter_property l_b) -> bool {
                      return l_a.stride_right > l_b.stride_right;
                    });
  std::stable_sort( m_iter_space->begin(), m_iter_space->end(), 
                    [&](iter_property l_a, iter_property l_b) -> bool {
                      return l_a.stride_out > l_b.stride_out;
                    });
  std::stable_sort( m_iter_space->begin(), m_iter_space->end(),
                    [&](iter_property l_a, iter_property l_b) -> bool {
                      return l_a.dim_type < l_b.dim_type;
                    });

  //fuse iters with the same dimension type and contiguous storage
  std::vector<iter_property>::iterator l_next;
  for( std::vector<iter_property>::iterator l_it = m_iter_space->begin(); l_it < m_iter_space->end() - 1; l_it = l_next ){
    l_next = l_it + 1;
    if( l_it->dim_type  == l_next->dim_type ){
      int64_t l_size = l_next->size; 
      if( l_next->stride_left    * l_size == l_it->stride_left    && 
          l_next->stride_right   * l_size == l_it->stride_right   && 
          l_next->stride_out     * l_size == l_it->stride_out     &&
          l_next->stride_out_aux * l_size == l_it->stride_out_aux    ){
        l_it->size           *= l_size;
        l_it->stride_left    /= l_size;
        l_it->stride_right   /= l_size;
        l_it->stride_out     /= l_size;
        l_it->stride_out_aux /= l_size;
        m_iter_space->erase(l_next);
        l_next = l_it;
      }
    }
  }
}

void einsum_ir::binary::ContractionOptimizer::remove_empty_iters(){
    std::vector<iter_property>::iterator l_it;
    l_it = std::remove_if( m_iter_space->begin(), 
                           m_iter_space->end(), 
                           [](const iter_property & l_iter) {
                              return l_iter.size == 1 && l_iter.exec_type != exec_t::PRIM;
                           });
    m_iter_space->erase(l_it, m_iter_space->end());
}


void einsum_ir::binary::ContractionOptimizer::find_iters_with_stride( std::vector<iter_property>::iterator & o_iter_left,
                                                                      std::vector<iter_property>::iterator & o_iter_right,
                                                                      std::vector<iter_property>::iterator & o_iter_out,
                                                                      int64_t i_stride ){
  
  o_iter_left = m_iter_space->end();
  o_iter_right = m_iter_space->end();
  o_iter_out = m_iter_space->end();

  std::vector<iter_property>::iterator l_it;
  for( l_it = m_iter_space->begin(); l_it < m_iter_space->end(); l_it++ ){
    if( l_it->stride_left == i_stride ){
      o_iter_left = l_it;
    }
    if( l_it->stride_right == i_stride ){
      o_iter_right = l_it;
    }
    if( l_it->stride_out == i_stride ){
      o_iter_out = l_it;
    }
  }
}

void einsum_ir::binary::ContractionOptimizer::find_iter_with_dimtype( std::vector<iter_property>::iterator & o_iter,
                                                                      dim_t i_dim_type ){
  o_iter = m_iter_space->end();
  int64_t l_min_stride = 0x7FFFFFFFFFFFFFFF;
  std::vector<iter_property>::iterator l_it;
  for( l_it = m_iter_space->begin(); l_it < m_iter_space->end(); l_it++ ){
    int64_t l_stride = l_it->stride_left + l_it->stride_right + l_it->stride_out;
    if(    l_it->dim_type == i_dim_type
        && l_stride < l_min_stride ){
      o_iter = l_it;
    }
  }
}

void einsum_ir::binary::ContractionOptimizer::get_size_all_m_n( int64_t & o_size_m,
                                                                int64_t & o_size_n ){
  std::vector<iter_property>::iterator l_it;
  o_size_m = 1;
  o_size_n = 1;
  for( l_it = m_iter_space->begin(); l_it < m_iter_space->end(); l_it++ ){
    if( l_it->dim_type == dim_t::M ){
      o_size_m *= l_it->size;
    }
    if( l_it->dim_type == dim_t::N ){
      o_size_n *= l_it->size;
    }
  }
}

void einsum_ir::binary::ContractionOptimizer::set_kernel_targets_heuristic( int64_t * i_potential_kernel_size,
                                                                            int64_t * io_kernel_targets ){
  //adapt all kernel target sizes depending on what is possible e.g. small k kernel -> choose bigger m target
  if( i_potential_kernel_size[ PRIM_C ] > 1 ){
    io_kernel_targets[ PRIM_C  ] = i_potential_kernel_size[ PRIM_C ];
  }
  if( i_potential_kernel_size[ PRIM_K ] < io_kernel_targets[ PRIM_K ] ){
    io_kernel_targets[ PRIM_BR ] *= io_kernel_targets[ PRIM_K ] / i_potential_kernel_size[ PRIM_K ];
    io_kernel_targets[ PRIM_K  ]  = i_potential_kernel_size[ PRIM_K ];
  }
  if( i_potential_kernel_size[ PRIM_BR ] < io_kernel_targets[ PRIM_BR ] ){
    io_kernel_targets[ PRIM_M  ] *= io_kernel_targets[ PRIM_BR ] / i_potential_kernel_size[ PRIM_BR ];
    io_kernel_targets[ PRIM_BR ]  = i_potential_kernel_size[ PRIM_BR ];
  }
  if( i_potential_kernel_size[PRIM_M] < io_kernel_targets[ PRIM_M ] ){
    io_kernel_targets[ PRIM_N ] *= io_kernel_targets[ PRIM_M ] / i_potential_kernel_size[ PRIM_M ];
    io_kernel_targets[ PRIM_M ]  = i_potential_kernel_size[ PRIM_M ];
  }
  if( i_potential_kernel_size[ PRIM_N ] < io_kernel_targets[ PRIM_N ] ){
    io_kernel_targets[ PRIM_N  ] = i_potential_kernel_size[ PRIM_N ];
  }


  //reduce kernel targets when parallelism is low
  int64_t l_size_m, l_size_n;
  get_size_all_m_n( l_size_m, l_size_n );
  int64_t l_possible_parallelism = l_size_m * l_size_n;
  while( l_possible_parallelism / (io_kernel_targets[PRIM_M] * io_kernel_targets[PRIM_N]) < m_num_threads &&
         io_kernel_targets[PRIM_M] * io_kernel_targets[PRIM_N] > 1 ){
    if(io_kernel_targets[PRIM_M] < io_kernel_targets[PRIM_N]){
      io_kernel_targets[PRIM_N] /= 2;
    }
    else{
      io_kernel_targets[PRIM_M] /= 2;
    }
  }
}


einsum_ir::err_t einsum_ir::binary::ContractionOptimizer::set_primitive_iters(){
  //The Algorithm consits of three steps:
  // 1. Find iterations of the iteration space that could be used for the kernel.
  // 2. Determine a good size for the potential kernel iterations depending on a heuristic or a performance model.
  // 3. Split the potential kernel iterations to the target size and mark them as a primitive .

  //-----------------------
  // Initial error handling
  //-----------------------

  if( *m_ktype_main != kernel_t::MADD ){
    return err_t::COMPILATION_FAILED;
  }

  //allocate extra memory for kernel
  m_iter_space->reserve( m_iter_space->size() + 5 ); 

  // move complex dimension to extra data structure
  std::vector<iter_property>::iterator l_complex_iter;
  iter_property l_complex_iter_prop; 
  find_iter_with_dimtype( l_complex_iter, dim_t::CPX );
  if( l_complex_iter != m_iter_space->end() ){
    l_complex_iter_prop = *l_complex_iter;
    m_iter_space->erase( l_complex_iter );
    if( l_complex_iter_prop.size != 2 ){
      return err_t::INVALID_CPX_DIM;
    }

    //check that only one complex dimension exists
    find_iter_with_dimtype( l_complex_iter, dim_t::CPX );
    if( l_complex_iter != m_iter_space->end() ){
      return err_t::INVALID_CPX_DIM;
    }
  }

  //----------------------------------------------
  // Variable definition for the optimization pass
  //----------------------------------------------

  //packing variables
  bool l_packing_left = false;
  bool l_packing_right = false;
  std::vector<iter_property>::iterator l_extra_packing_iter_left  = m_iter_space->end();
  std::vector<iter_property>::iterator l_extra_packing_iter_right = m_iter_space->end();
  
  //kernel variables
  //enum                                                           {        PRIM_BR = 0,         PRIM_C  = 1,         PRIM_M  = 2,        PRIM_N  = 3,          PRIM_K  = 4};
  dim_t                                           l_iter_dim_t[] = {           dim_t::K,            dim_t::C,            dim_t::M,            dim_t::N,            dim_t::K};
  bool                                         l_iter_required[] = {              false,               false,                true,                true,                true};
  int64_t                                     l_kernel_targets[] = {                  1,                   1,          m_target_m,          m_target_n,          m_target_k};
  int64_t                              l_potential_kernel_size[] = {                  1,                   1,                   1,                   1,                   1};
  std::vector<iter_property>::iterator l_potential_kernel_iter[] = {m_iter_space->end(), m_iter_space->end(), m_iter_space->end(), m_iter_space->end(), m_iter_space->end()};

  //-----------------------------------------
  // Step 1: Find potential kernel iterations
  //-----------------------------------------

  //variables definition for step 1
  int64_t l_req_stride = 1;
  bool l_transpose_a = false;
  bool l_transpose_b = false;
  std::vector<iter_property>::iterator l_small_stride_left;
  std::vector<iter_property>::iterator l_small_stride_right;
  std::vector<iter_property>::iterator l_small_stride_out;

  //get stride one dimensions
  find_iters_with_stride( l_small_stride_left,
                          l_small_stride_right,
                          l_small_stride_out,
                          l_req_stride );


  //find possible C kernel dimension
  if(    l_small_stride_out != m_iter_space->end()
      && l_small_stride_out->dim_type == dim_t::C ){
    l_potential_kernel_iter[ PRIM_C ] = l_small_stride_out;
    l_potential_kernel_size[ PRIM_C ] = l_small_stride_out->size;
    l_req_stride *= l_small_stride_out->size;

    if(l_small_stride_left != l_small_stride_out){
      l_packing_left = true;
      l_extra_packing_iter_left = l_small_stride_left;
    }
    if(l_small_stride_right != l_small_stride_out){
      l_packing_right = true;
      l_extra_packing_iter_right = l_small_stride_right;
    }
  }

  //get dimensions with a stride matching the required stride
  find_iters_with_stride( l_small_stride_left,
                          l_small_stride_right,
                          l_small_stride_out,
                          l_req_stride );
  
  //in case of packing use the extra dimension for kernel search
  if( l_packing_left ){
    l_small_stride_left = l_extra_packing_iter_left;
    l_extra_packing_iter_left = m_iter_space->end();
  }
  if( l_packing_right ){
    l_small_stride_right = l_extra_packing_iter_right;
    l_extra_packing_iter_right = m_iter_space->end();
  }

  //find possible M kernel dimension
  if(    l_small_stride_out != m_iter_space->end()
      && l_small_stride_out->dim_type == dim_t::M ){
    l_potential_kernel_iter[ PRIM_M ] = l_small_stride_out;
    l_potential_kernel_size[ PRIM_M ] = l_small_stride_out->size;
    l_transpose_a = l_small_stride_out == l_small_stride_left ? false : true;
  }

  //find possible N kernel dimension
  if(    l_small_stride_left != m_iter_space->end()
      && l_small_stride_left->dim_type == dim_t::N ){
    l_potential_kernel_iter[ PRIM_N ] = l_small_stride_left;
    l_potential_kernel_size[ PRIM_N ] = l_small_stride_left->size;
  }
  else{
    l_transpose_b = true;
    std::vector<iter_property>::iterator l_iter;
    find_iter_with_dimtype( l_iter, dim_t::N );
    if( l_iter != m_iter_space->end() ){
      l_potential_kernel_iter[ PRIM_N ] = l_iter;
      l_potential_kernel_size[ PRIM_N ] = l_iter->size;
    }
  }

  //find possible K kernel dimension
  if(    !l_transpose_b 
      && l_small_stride_right != m_iter_space->end() ){
    l_potential_kernel_iter[ PRIM_K ] = l_small_stride_right;
    l_potential_kernel_size[ PRIM_K ] = l_small_stride_right->size;

    if(    l_transpose_a 
        && l_small_stride_left != l_small_stride_right ){
      l_packing_left = true;
      l_extra_packing_iter_right = m_iter_space->end();
    }
  }
  else if( l_transpose_b ){
    if(    l_transpose_a 
        && l_small_stride_left != m_iter_space->end() ){
      l_potential_kernel_iter[ PRIM_K ] = l_small_stride_left;
      l_potential_kernel_size[ PRIM_K ] = l_small_stride_left->size;
    }
    else{
      std::vector<iter_property>::iterator l_iter;
      find_iter_with_dimtype( l_iter, dim_t::K );
      if( l_iter != m_iter_space->end() ){
        l_potential_kernel_iter[ PRIM_K ] = l_iter;
        l_potential_kernel_size[ PRIM_K ] = l_iter->size;
      }
    }
  }

  //find possible BR kernel dimension
  if(    m_br_gemm_support 
      && l_potential_kernel_size[ PRIM_C ] > 1){
    //if an extra packing dimension is available use it as BR kernel
    if( l_extra_packing_iter_left != m_iter_space->end() &&
        l_extra_packing_iter_left->dim_type == dim_t::K ){
      l_potential_kernel_iter[ PRIM_BR ] = l_extra_packing_iter_left;
      l_potential_kernel_size[ PRIM_BR ] = l_extra_packing_iter_left->size;
      l_extra_packing_iter_left = m_iter_space->end();
    }
    //if no extra packing dimension is available search for another K dimension
    else if( l_potential_kernel_iter[PRIM_K] != m_iter_space->end()){   
      l_potential_kernel_iter[PRIM_K]->dim_type = dim_t::UNDEFINED_DIM;
      std::vector<iter_property>::iterator l_iter;
      find_iter_with_dimtype( l_iter, dim_t::K );
      l_potential_kernel_iter[PRIM_K]->dim_type = dim_t::K;
      if( l_iter != m_iter_space->end() ){
        l_potential_kernel_iter[ PRIM_BR ] = l_iter;
        l_potential_kernel_size[ PRIM_BR ] = l_iter->size;
      }
    }
  }

  // pack input tensor if strides are a mulitple of a large power of 2 (2048)
  // optimization to prevent cache misses caused by mapping to the same cache lane
  if(    l_potential_kernel_size[ PRIM_K ] > 1
      && l_potential_kernel_iter[ PRIM_K ]->stride_left % 2048 == 0 ){
    l_packing_left = true;
  }
  if(    l_potential_kernel_size[ PRIM_M ] > 1
      && l_potential_kernel_iter[ PRIM_M ]->stride_left % 2048 == 0 ){
    l_packing_left = true;
  }

  //---------------------------------
  // Step 2: Determine kernel targets
  //---------------------------------

  //addapts the kernel targets depending on the potential kernel size
  set_kernel_targets_heuristic( l_potential_kernel_size, l_kernel_targets );
  //TODO use a performance model to determine the kernel targets

  //------------------------------------------
  // Step 3: Split potential kernel iterations
  //------------------------------------------

  //add CPX dimensions and set kernel type
  if( l_complex_iter_prop.dim_type == dim_t::CPX ){
    m_iter_space->push_back(l_complex_iter_prop);
    if( l_kernel_targets[PRIM_C] > 1 ){
      *m_ktype_main = kernel_t::CPX_PACKED_MADD;
    }
    else{
      *m_ktype_main = kernel_t::CPX_MADD;
    }
  }
  else{
    if( l_kernel_targets[PRIM_C] > 1 ){
      *m_ktype_main = kernel_t::PACKED_MADD;
    }
    else if( l_kernel_targets[PRIM_BR] > 1 ){
      *m_ktype_main = kernel_t::BR_MADD;
    }
  }

  //TODO use better split
  //TODO use another execution type for better identification
  //add extra packing dimension if required
  if( l_extra_packing_iter_left != m_iter_space->end() ){
    split_iter( l_extra_packing_iter_left,
                8,
                -1,
                exec_t::SEQ );
  }

  //create kernel from potential kernel dimensions
  for( int64_t l_prim_id = 0; l_prim_id < 5; l_prim_id++ ){
    if( l_kernel_targets[l_prim_id] > 1 ){
      split_iter( l_potential_kernel_iter[l_prim_id],
                  l_kernel_targets[l_prim_id],
                  -1, //add to the end
                  exec_t::PRIM );
    }
    else if( l_iter_required[l_prim_id] ){
      //std::cout << "adding empy" << std::endl;
      iter_property l_new_iter;

      l_new_iter.dim_type  = l_iter_dim_t[l_prim_id];
      l_new_iter.exec_type = exec_t::PRIM;
      l_new_iter.size      = 1;
      m_iter_space->push_back(l_new_iter);

    }
  }

  //set packing strides
  if( l_packing_left || l_packing_right ){
    int64_t l_next_id = -4;
    std::vector<int64_t> l_packing_order;
    if( *m_ktype_main == kernel_t::PACKED_MADD ||
        *m_ktype_main == kernel_t::CPX_PACKED_MADD ){
      l_packing_order.push_back(l_next_id--);
    }
    l_packing_order.push_back(-3);
    l_packing_order.push_back(-1);
    l_packing_order.push_back(-2);
    if( *m_ktype_main == kernel_t::BR_MADD ){
      l_packing_order.push_back(l_next_id--);
    }
    if( *m_ktype_main == kernel_t::CPX_MADD ||
        *m_ktype_main == kernel_t::CPX_PACKED_MADD){
      l_packing_order.push_back(l_next_id--);
    }
    if( l_extra_packing_iter_left != m_iter_space->end() ){
      l_packing_order.push_back(l_next_id--);
    }
    if( l_extra_packing_iter_right != m_iter_space->end() ){
      l_packing_order.push_back(l_next_id--);
    }

    int64_t l_stride_left  = 1;
    int64_t l_stride_right = 1;
    for( size_t l_id = 0; l_id < l_packing_order.size(); l_id++ ){
      int64_t l_packing_id = l_packing_order[l_id];
      std::vector<iter_property>::iterator l_iter = m_iter_space->end() + l_packing_id;
      if( l_iter->stride_left > 0 && l_packing_left ){
        l_iter->packing_stride_left = l_iter->stride_left;
        l_iter->stride_left = l_stride_left;
        l_stride_left *= l_iter->size;
      }
      if( l_iter->stride_right > 0 && l_packing_right ){
        l_iter->packing_stride_right = l_iter->stride_right;
        l_iter->stride_right = l_stride_right;
        l_stride_right *= l_iter->size;
      }
    }
  }
  
  return err_t::SUCCESS;
}

void einsum_ir::binary::ContractionOptimizer::reorder_and_parallelize_iters(){
  //move primitive iterations to another data structure
  std::vector<iter_property> l_kernel_iters;
  std::vector<iter_property>::iterator l_it = m_iter_space->begin();
  while( l_it < m_iter_space->end() ){
    if(    l_it->exec_type == exec_t::PRIM 
        || l_it->packing_stride_left > 0 
        || l_it->packing_stride_right > 0 ){
      l_kernel_iters.push_back(*l_it);
      l_it = m_iter_space->erase(l_it);
    }
    else{
      l_it++;
    }
  }

  int64_t l_kernel_size_out = 1;
  for( l_it = l_kernel_iters.begin(); l_it < l_kernel_iters.end(); l_it++ ){
    if( l_it->dim_type != dim_t::K ){
      l_kernel_size_out *= l_it->size;
    }
  }

  //use about half of the L2 cache for C blocking (A and B tend to be a lot smaller because of SFC blocking in M and N
  int64_t l_target_thread_tasks = m_l2_cache_size / 2 / (l_kernel_size_out * m_num_bytes_scalar_out );
  int64_t l_target_parallel = m_num_threads * l_target_thread_tasks;

  //simple heuristic to determine parallel targets
  int64_t l_target_parallel_c = 1;
  int64_t l_target_parallel_m, l_target_parallel_n;
  get_size_all_m_n( l_target_parallel_m, l_target_parallel_n );
  while( l_target_parallel_m * l_target_parallel_n > l_target_parallel ){
    if( l_target_parallel_m >= l_target_parallel_n ){
      l_target_parallel_m /= 2;
    }
    else{
      l_target_parallel_n /= 2;
    }
  }
  l_target_parallel_c = l_target_parallel / (l_target_parallel_m * l_target_parallel_n);

  //add parallel dimension
  std::vector<iter_property> l_parallel_iters;
  move_iters_until( &l_parallel_iters, 
                    l_target_parallel_n,
                    dim_t::N,
                    exec_t::SFC);
  move_iters_until( &l_parallel_iters, 
                    l_target_parallel_m,
                    dim_t::M,
                    exec_t::SFC);
  move_iters_until( &l_parallel_iters, 
                    l_target_parallel_c,
                    dim_t::C,
                    exec_t::OMP);

  //sort remaining dimensions by sum of strides
  std::sort( m_iter_space->begin(), m_iter_space->end(), 
             [&](iter_property l_a, iter_property l_b) -> bool {
                int64_t l_stride_a  = l_a.stride_left + l_a.stride_right + l_a.stride_out;
                int64_t l_stride_b = l_b.stride_left + l_b.stride_right + l_b.stride_out; 
                return l_stride_a > l_stride_b;
             });
  
  //add iterations from local data structures
  m_iter_space->insert(m_iter_space->end(), l_parallel_iters.begin(), l_parallel_iters.end() );
  m_iter_space->insert(m_iter_space->end(), l_kernel_iters.begin(), l_kernel_iters.end() );
}

void einsum_ir::binary::ContractionOptimizer::move_iters_until( std::vector<iter_property> * i_dest_iters,
                                                                int64_t                      i_target_size,
                                                                dim_t                        i_dim_type, 
                                                                exec_t                       i_new_exec_t ){

  int64_t l_target_remaining = i_target_size;
  std::vector<iter_property>::iterator l_it;
  find_iter_with_dimtype( l_it, i_dim_type );
  while(     l_target_remaining > 1
          && l_it != m_iter_space->end() ){
    //if size is smalle than target size split it
    int64_t l_size_iter  = l_it->size;
    if(l_size_iter > l_target_remaining){
      split_iter( l_it,
                  l_target_remaining,
                  -1, 
                  i_new_exec_t );
      l_it = m_iter_space->end() - 1;
    }

    //add iter to destination
    iter_property l_new_iter = *l_it;
    l_new_iter.exec_type = i_new_exec_t;
    i_dest_iters->push_back(l_new_iter);
    m_iter_space->erase(l_it);
    l_target_remaining /= l_size_iter;
  }
}

int64_t einsum_ir::binary::ContractionOptimizer::move_all_iters( std::vector<iter_property> * i_source_iters,
                                                                 std::vector<iter_property> * i_dest_iters,
                                                                 exec_t                       i_new_exec_t ){
  int64_t l_size_all = 1;
  while( i_source_iters->size() > 0 ){
    std::vector<iter_property>::iterator l_it = i_source_iters->end() - 1;
    l_size_all *= move_iter( l_it,
                             i_source_iters,
                             i_dest_iters,
                             0,
                             i_new_exec_t );
  }
  return l_size_all;
}

void einsum_ir::binary::ContractionOptimizer::split_iter( std::vector<iter_property>::iterator i_iteration,
                                                          int64_t                              i_target_size,
                                                          int64_t                              i_new_iter_pos, 
                                                          exec_t                               i_new_exec_t ){
  int64_t l_split = find_split(i_iteration->size, i_target_size );

  iter_property l_new_iter = *i_iteration;
  l_new_iter.exec_type = i_new_exec_t;
  l_new_iter.size = l_split;
  if( i_new_iter_pos >= 0 ){
    m_iter_space->insert(m_iter_space->begin() + i_new_iter_pos, l_new_iter );
  }
  else{
    m_iter_space->insert(m_iter_space->end() + 1 + i_new_iter_pos, l_new_iter );
  }

  i_iteration->size           /= l_split;
  i_iteration->stride_left    *= l_split;
  i_iteration->stride_right   *= l_split;
  i_iteration->stride_out_aux *= l_split;
  i_iteration->stride_out     *= l_split;                        
}

void einsum_ir::binary::ContractionOptimizer::add_empty_iter( std::vector<iter_property> * i_dest_iters,
                                                              int64_t                      i_new_iter_pos, 
                                                              dim_t                        i_new_dim_t,
                                                              exec_t                       i_new_exec_t ){

  iter_property l_empty_iter;

  l_empty_iter.dim_type = i_new_dim_t;
  l_empty_iter.exec_type = i_new_exec_t;
  l_empty_iter.size = 1;
  l_empty_iter.stride_left = 0;
  l_empty_iter.stride_right = 0;
  l_empty_iter.stride_out = 0;  
  l_empty_iter.stride_out_aux = 0;

  i_dest_iters->insert(i_dest_iters->begin() + i_new_iter_pos, l_empty_iter );                                                       
}

int64_t einsum_ir::binary::ContractionOptimizer::move_iter( std::vector<iter_property>::iterator i_iteration,
                                                            std::vector<iter_property> *         i_source_iters,
                                                            std::vector<iter_property> *         i_dest_iters,
                                                            int64_t                              i_new_iter_pos, 
                                                            exec_t                               i_new_exec_t ){
  
  i_dest_iters->insert(i_dest_iters->begin() + i_new_iter_pos, *i_iteration );
  i_dest_iters->at(i_new_iter_pos).exec_type = i_new_exec_t;
  i_source_iters->erase( i_iteration );
    
  return  i_iteration->size;

  return 1;                                                         
}

int64_t einsum_ir::binary::ContractionOptimizer::find_split( int64_t i_dim_size,
                                                             int64_t i_target_size
                                                           ){
  //factorization of number
  int64_t l_best_factor = i_dim_size;
  double l_best_distance = std::abs(std::log((double)i_dim_size/i_target_size));

  for(int64_t l_i = 1; l_i <= (int64_t)std::sqrt(i_dim_size); l_i++){
    if(i_dim_size % l_i == 0){
      double l_distance_i = std::abs(std::log((double)l_i/i_target_size));
      if(l_best_distance > l_distance_i){
        l_best_factor = l_i;
        l_best_distance = l_distance_i;
      }
      int64_t l_other = i_dim_size / l_i;
      l_distance_i = std::abs(std::log((double)l_other/i_target_size));
      if(l_best_distance > l_distance_i){
        l_best_factor = l_other;
        l_best_distance = l_distance_i;
      }
    }
  }
  
  return l_best_factor;
}