#include "ContractionOptimizer.h"
#include <algorithm>
#include <cmath>

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

  // sort iters depeding on strides, exec_types and dim_types
  sort_iters();

  // fuse iters if possible
  fuse_iters();

  // move iters to internal data structure
  move_iters_to_internal();

  // find and add the Kernel
  add_kernel();

  // reoders and adds all remaining iters
  reorder_iters();
}

void einsum_ir::binary::ContractionOptimizer::sort_iters(){
  
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
  std::stable_sort( m_iter_space->begin(), m_iter_space->end(), 
                    [&](iter_property l_a, iter_property l_b) -> bool {
                      return l_a.exec_type < l_b.exec_type;
                    });
}

void einsum_ir::binary::ContractionOptimizer::fuse_iters(){
  std::vector<iter_property>::iterator l_next;
  for( std::vector<iter_property>::iterator l_it = m_iter_space->begin(); l_it < m_iter_space->end() - 1; l_it = l_next ){
    l_next = l_it + 1;
    if( l_it->dim_type  == l_next->dim_type && 
        l_it->exec_type == l_next->exec_type ){
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
  for( std::vector<iter_property>::iterator l_it = m_iter_space->begin(); l_it < m_iter_space->end(); l_it++ ){
    if( l_it->size      == 1 &&
        l_it->exec_type != exec_t::PRIM ){
      l_it--;
      m_iter_space->erase( l_it + 1 );
    }
  }
}

void einsum_ir::binary::ContractionOptimizer::move_iters_to_internal(){
  m_size_all_m = 1;
  m_size_all_n = 1;
  m_free_iters.resize(5);
  std::vector<iter_property>::iterator l_it = m_iter_space->begin();
  while( l_it < m_iter_space->end() ){
    if( l_it->exec_type == exec_t::SEQ ){
      if( l_it->dim_type == dim_t::M ){
        m_size_all_m *= l_it->size;
      }
      if( l_it->dim_type == dim_t::N ){
        m_size_all_n *= l_it->size;
      }
      m_free_iters[l_it->dim_type].push_back(*l_it);
      l_it--;
      m_iter_space->erase(l_it + 1);
    }
    l_it++;
  }
}

void einsum_ir::binary::ContractionOptimizer::add_kernel(){
  enum {
    PRIM_BR = 0,
    PRIM_C  = 1,
    PRIM_M  = 2,
    PRIM_N  = 3,
    PRIM_K  = 4
  };

  //find possible kernel dimensions
  bool l_transpose_a = false;
  bool l_transpose_b = false;
  int64_t l_req_stride = 1;
  int64_t l_potential_kernel_size[] = {1,1,1,1,1};
  std::vector<iter_property>::iterator l_iteration;
  std::vector<iter_property>::iterator l_potential_kernel_iter[] = { m_free_iters[ dim_t::K ].end(),
                                                                     m_free_iters[ dim_t::C ].end(),
                                                                     m_free_iters[ dim_t::M ].end(),
                                                                     m_free_iters[ dim_t::N ].end(),
                                                                     m_free_iters[ dim_t::K ].end() };  
  //find possible C kernel dimension
  if(    m_free_iters[ dim_t::C ].size() > 0 
      && m_packed_gemm_support  == true      ) {
    l_iteration = m_free_iters[ dim_t::C ].end() - 1;
    if(    l_iteration->stride_left  == l_req_stride
        && l_iteration->stride_right == l_req_stride
        && l_iteration->stride_out   == l_req_stride ){
      l_potential_kernel_iter[ PRIM_C ] = l_iteration;
      l_potential_kernel_size[ PRIM_C ] = l_iteration->size;
      l_req_stride *= l_iteration->size;
    }
  }
  
  //find possible M kernel dimension
  if( m_free_iters[ dim_t::M ].size() > 0 ) {
    l_iteration = m_free_iters[ dim_t::M ].end() - 1;
    if( l_iteration->stride_out  == l_req_stride    ){
      l_potential_kernel_iter[ PRIM_M ] = l_iteration;
      l_potential_kernel_size[ PRIM_M ] = l_iteration->size;
      l_transpose_a = l_iteration->stride_left == l_req_stride ? false : true;
    }
  }

  //find possible N kernel dimension
  if( m_free_iters[ dim_t::N ].size() > 0 ){
    l_iteration = m_free_iters[ dim_t::N ].end() - 1;
    l_potential_kernel_iter[ PRIM_N ] = l_iteration;
    l_potential_kernel_size[ PRIM_N ] = l_iteration->size;
    l_transpose_b = l_iteration->stride_right == l_req_stride ? true : false;
  }

  //find possible K kernel dimension
  if( m_free_iters[ dim_t::K ].size() > 0 ) {
    l_iteration = m_free_iters[ dim_t::K ].end() - 1;
    if( l_transpose_a && l_transpose_b ){
      while( l_iteration >= m_free_iters[ dim_t::K ].begin() + 1 && 
            l_iteration->stride_left != l_req_stride ){
        l_iteration--;
      }
    }
    if( ( l_iteration->stride_left  == l_req_stride || !l_transpose_a ) &&
        ( l_iteration->stride_right == l_req_stride ||  l_transpose_b )    ){
      l_potential_kernel_iter[ PRIM_K ] = l_iteration;
      l_potential_kernel_size[ PRIM_K ] = l_iteration->size; 
    }
  }

  //find possible BR kernel dimension
  if( m_free_iters[ dim_t::K ].size() > 1 &&
      m_br_gemm_support == true                ){
    l_iteration = m_free_iters[ dim_t::K ].end() - 2;
    l_potential_kernel_iter[ PRIM_BR ] = l_iteration;
    l_potential_kernel_size[ PRIM_BR ] = l_iteration->size;
  }


  //adapt all kernel target sizes depending on what is possible e.g. small k kernel -> choose bigger m target
  int64_t l_kernel_targets[] = {1, 1, m_target_m, m_target_n, m_target_k};
  if( l_potential_kernel_size[ PRIM_C ] > 1 ){
    l_kernel_targets[ PRIM_C  ] = l_potential_kernel_size[ PRIM_C ];
    l_potential_kernel_size[ PRIM_BR ] = 1;
  }
  if( l_potential_kernel_size[ PRIM_K ] < l_kernel_targets[ PRIM_K ] ){
    l_kernel_targets[ PRIM_BR ] *= l_kernel_targets[ PRIM_K ] / l_potential_kernel_size[ PRIM_K ];
    l_kernel_targets[ PRIM_K  ]  = l_potential_kernel_size[ PRIM_K ];
  }
  if( l_potential_kernel_size[ PRIM_BR ] < l_kernel_targets[ PRIM_BR ] ){
    l_kernel_targets[ PRIM_M  ] *= l_kernel_targets[ PRIM_BR ] / l_potential_kernel_size[ PRIM_BR ];
    l_kernel_targets[ PRIM_BR ]  = l_potential_kernel_size[ PRIM_BR ];
  }
  if( l_potential_kernel_size[PRIM_M] < l_kernel_targets[ PRIM_M ] ){
    l_kernel_targets[ PRIM_N ] *= l_kernel_targets[ PRIM_M ] / l_potential_kernel_size[ PRIM_M ];
    l_kernel_targets[ PRIM_M ]  = l_potential_kernel_size[ PRIM_M ];
  }
  if( l_potential_kernel_size[ PRIM_N ] < l_kernel_targets[ PRIM_N ] ){
    l_kernel_targets[ PRIM_N  ] = l_potential_kernel_size[ PRIM_N ];
  }


  //reduce kernel targets when parallelism is low
  int64_t l_possible_parallelism = m_size_all_m * m_size_all_n;
  while( l_possible_parallelism / (l_kernel_targets[PRIM_M] * l_kernel_targets[PRIM_N]) < m_num_threads &&
         l_kernel_targets[PRIM_M] * l_kernel_targets[PRIM_N] > 1 ){
    if(l_kernel_targets[PRIM_M] < l_kernel_targets[PRIM_N]){
      l_kernel_targets[PRIM_N] /= 2;
    }
    else{
      l_kernel_targets[PRIM_M] /= 2;
    }
  }

  //create kernel from potential kernel dimensions
  int64_t l_kernel_sizes[] = {1, 1, 1, 1, 1};
  bool l_iter_required[] = {false, false, true, true, true};
  dim_t l_iter_dim_t[] = {dim_t::K, dim_t::C, dim_t::M, dim_t::N, dim_t::K};
  for( int64_t l_prim_id = 4; l_prim_id >= 0; l_prim_id-- ){
    if( l_kernel_targets[l_prim_id] > 1 ){
      l_kernel_sizes[l_prim_id] = split_iter( l_potential_kernel_iter[l_prim_id],
                                              &m_free_iters[l_iter_dim_t[l_prim_id]],
                                              m_iter_space,
                                              l_kernel_targets[l_prim_id],
                                              0,
                                              exec_t::PRIM );
    }
    else if( l_iter_required[l_prim_id] ){
      add_empty_iter( m_iter_space, 0, l_iter_dim_t[l_prim_id], exec_t::PRIM );
    }
  }

  //add CPX dimensions and set kernel type
  if( m_free_iters[ dim_t::CPX ].size() > 0 ){
    move_all_iters( &m_free_iters[ dim_t::CPX ],
                    m_iter_space,
                    exec_t::PRIM );
    if( l_kernel_targets[PRIM_C] > 1 ){
      *m_ktype_main = kernel_t::CPX_PACKED_MADD;
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
  
  m_size_all_m /= l_kernel_sizes[PRIM_M];
  m_size_all_n /= l_kernel_sizes[PRIM_N];

  //use about half of the L2 cache for C blocking (A and B tend to be a lot smaller because of SFC blocking in M and )
  int64_t l_target_thread_tasks = m_l2_cache_size / 2 / (l_kernel_sizes[PRIM_M] * l_kernel_sizes[PRIM_N] * m_num_bytes_scalar_out );
  m_target_parallel = m_num_threads * l_target_thread_tasks;
}

void einsum_ir::binary::ContractionOptimizer::reorder_iters(){
  //determine parallel targets similarly to kernel targets
  int64_t l_target_parallel_m = m_size_all_m;
  int64_t l_target_parallel_n = m_size_all_n;
  int64_t l_target_parallel_c = 1;

  while( l_target_parallel_m * l_target_parallel_n > m_target_parallel ){
    if( l_target_parallel_m >= l_target_parallel_n ){
      l_target_parallel_m /= 2;
    }
    else{
      l_target_parallel_n /= 2;
    }
  }
  l_target_parallel_c = m_target_parallel / (l_target_parallel_m * l_target_parallel_n);

  //add parallel dimension
  move_iters_until( &m_free_iters[dim_t::N],
                    m_iter_space, 
                    l_target_parallel_n,
                    exec_t::SFC);
  move_iters_until( &m_free_iters[dim_t::M],
                    m_iter_space, 
                    l_target_parallel_m,
                    exec_t::SFC);
  move_iters_until( &m_free_iters[dim_t::C],
                    m_iter_space, 
                    l_target_parallel_c,
                    exec_t::OMP);


  //add remaining dimensions
  move_all_iters( &m_free_iters[dim_t::K],
                  m_iter_space,
                  exec_t::SEQ );
  move_all_iters( &m_free_iters[dim_t::M],
                  m_iter_space,
                  exec_t::SEQ );
  move_all_iters( &m_free_iters[dim_t::N],
                  m_iter_space,
                  exec_t::SEQ );
  move_all_iters( &m_free_iters[dim_t::C], 
                  m_iter_space,
                  exec_t::SEQ );
}

int64_t einsum_ir::binary::ContractionOptimizer::move_iters_until( std::vector<iter_property> * i_source_iters,
                                                                   std::vector<iter_property> * i_dest_iters,
                                                                   int64_t                      i_target_size,
                                                                   exec_t                       i_new_exec_t ){
  int64_t l_size_all = 1;
  int64_t l_target_size = i_target_size;
  while( i_source_iters->size() > 0 && 
         l_target_size > 1 ){
    std::vector<iter_property>::iterator l_it = i_source_iters->end() - 1;
    int64_t l_size_iter  = l_it->size;
    int64_t l_size_split = split_iter( l_it,
                                       i_source_iters,
                                       i_dest_iters,
                                       l_target_size,
                                       0,
                                       i_new_exec_t );

    l_size_all *= l_size_iter;
    l_target_size = i_target_size / l_size_all;
    if( l_size_split < l_size_iter ){
      l_target_size = 0;
    }
  }

  return l_size_all;
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

int64_t einsum_ir::binary::ContractionOptimizer::split_iter( std::vector<iter_property>::iterator i_iteration,
                                                             std::vector<iter_property> *         i_source_iters,
                                                             std::vector<iter_property> *         i_dest_iters,
                                                             int64_t                              i_target_size,
                                                             int64_t                              i_new_iter_pos, 
                                                             exec_t                               i_new_exec_t ){

  int64_t l_split = find_split(i_iteration->size, i_target_size );

  i_dest_iters->insert(i_dest_iters->begin() + i_new_iter_pos, *i_iteration );
  i_dest_iters->at(i_new_iter_pos).exec_type = i_new_exec_t;
  i_dest_iters->at(0).size = l_split;

  i_iteration->size           /= l_split;
  i_iteration->stride_left    *= l_split;
  i_iteration->stride_right   *= l_split;
  i_iteration->stride_out_aux *= l_split;
  i_iteration->stride_out     *= l_split;

  if( i_iteration->size == 1 ){
      i_source_iters->erase( i_iteration );
  }
  return l_split;                                                         
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

  int64_t l_max_factor = i_dim_size / 2 < m_max_factor ? i_dim_size / 2 : m_max_factor;
  for(int64_t l_i = 2; l_i <= l_max_factor; l_i++){
    if(i_dim_size % l_i == 0){
      double l_distance_i = std::abs(std::log((double)l_i/i_target_size));
      if(l_best_distance > l_distance_i){
        l_best_factor = l_i;
        l_best_distance = l_distance_i;
      }
    }
  }
  
  return l_best_factor;
}