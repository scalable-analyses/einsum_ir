#include "ContractionOptimizer.h"
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::backend::ContractionOptimizer::init( std::vector< loop_property > * i_loops,
                                                     kernel_main_t                * i_ktype_main,
                                                     int64_t                        i_num_threads ){
  m_loops = i_loops;
  m_ktype_main = i_ktype_main;
  m_num_threads = i_num_threads;
}

void einsum_ir::backend::ContractionOptimizer::optimize(){
  // sort loops depeding on strides, exec_types and dim_types
  sortLoops();

  // fuse loops if possible
  fuseLoops();

  // move loops to internal data structure
  moveLoopsToInternal();

  // find and add the Kernel
  addKernel();

  // reodere loops and determine parallel loops
  reorderLoops();
}

void einsum_ir::backend::ContractionOptimizer::sortLoops(){
  
  std::stable_sort( m_loops->begin(), m_loops->end(), 
                    [&](loop_property l_a, loop_property l_b) -> bool {
                      return l_a.stride_right > l_b.stride_right;
                    });
  std::stable_sort( m_loops->begin(), m_loops->end(), 
                    [&](loop_property l_a, loop_property l_b) -> bool {
                      return l_a.stride_out > l_b.stride_out;
                    });
  std::stable_sort( m_loops->begin(), m_loops->end(),
                    [&](loop_property l_a, loop_property l_b) -> bool {
                      return l_a.dim_type < l_b.dim_type;
                    });
  std::stable_sort( m_loops->begin(), m_loops->end(), 
                    [&](loop_property l_a, loop_property l_b) -> bool {
                      return l_a.exec_type < l_b.exec_type;
                    });
}

void einsum_ir::backend::ContractionOptimizer::fuseLoops(){
  for( std::vector<loop_property>::iterator l_it = m_loops->begin() + 1; l_it < m_loops->end(); l_it++ ){
    std::vector<loop_property>::iterator l_other = l_it-1;
    if( l_it->dim_type  == l_other->dim_type && 
        l_it->exec_type == l_other->exec_type ){
      int64_t l_size = l_it->size; 
      if( l_it->stride_left    * l_size == l_other->stride_left    && 
          l_it->stride_right   * l_size == l_other->stride_right   && 
          l_it->stride_out     * l_size == l_other->stride_out     &&
          l_it->stride_out_aux * l_size == l_other->stride_out_aux    ){
        l_it->size *= l_other->size;
        m_loops->erase(l_other);
        l_it--;
      }
    }
  }
}

void einsum_ir::backend::ContractionOptimizer::moveLoopsToInternal(){
  m_size_all_m = 1;
  m_size_all_n = 1;
  m_free_loops.resize(4);
  std::vector<loop_property>::iterator l_it = m_loops->begin();
  while( l_it < m_loops->end() ){
    if( l_it->exec_type == exec_t::SEQ ){
      if( l_it->dim_type == dim_t::M ){
        m_size_all_m *= l_it->size;
      }
      if( l_it->dim_type == dim_t::N ){
        m_size_all_n *= l_it->size;
      }
      m_free_loops[l_it->dim_type].push_back(*l_it);
      m_loops->erase(l_it);
    }
    else{
      l_it++;
    }
  }
}

void einsum_ir::backend::ContractionOptimizer::addKernel(){
  typedef enum {
    PRIM_BR = 0,
    PRIM_M  = 1,
    PRIM_N  = 2,
    PRIM_K  = 3
  } gemm_prim_t;

  //find possible kernel dimensions
  int64_t l_potential_kernel_size[] = {1,1,1,1};
  std::vector<loop_property>::iterator l_potential_kernel_loop[] = { m_free_loops[ dim_t::K ].end(),
                                                                     m_free_loops[ dim_t::M ].end(),
                                                                     m_free_loops[ dim_t::N ].end(),
                                                                     m_free_loops[ dim_t::K ].end() };  
  //find possible M kernel dimension
  std::vector<loop_property>::iterator l_loop;
  l_loop = m_free_loops[ dim_t::M ].end() - 1;
  if( m_free_loops[ dim_t::M ].size() > 0 &&
      l_loop->stride_left == 1 && 
      l_loop->stride_out  == 1 ){
    l_potential_kernel_loop[ PRIM_M ] = l_loop;
    l_potential_kernel_size[ PRIM_M ] = l_loop->size;
  }
  else{
    push_empty_loop( &m_free_loops[ dim_t::M ], dim_t::M );
    l_potential_kernel_loop [PRIM_M ] = m_free_loops[ dim_t::M ].end() - 1;
  }

  //find possible N kernel dimension
  l_loop = m_free_loops[ dim_t::N ].end() - 1;
  if( m_free_loops[ dim_t::N ].size() > 0 ){
    l_potential_kernel_loop[ PRIM_N ] = l_loop;
    l_potential_kernel_size[ PRIM_N ] = l_loop->size;
  }
  else{
    push_empty_loop( &m_free_loops[ dim_t::N ], dim_t::N );
    l_potential_kernel_loop[ PRIM_N ] = m_free_loops[ dim_t::N ].end() - 1;
  }

  //find possible K kernel dimension
  l_loop = m_free_loops[ dim_t::K ].end() - 1;
  if( m_free_loops[ dim_t::K ].size() > 0 && 
      l_loop->stride_right == 1 ){
    l_potential_kernel_loop[ PRIM_K ] = l_loop;
    l_potential_kernel_size[ PRIM_K ] = l_loop->size; 
  }
  else{
    push_empty_loop( &m_free_loops[ dim_t::K ], dim_t::K );
    l_potential_kernel_loop[ PRIM_K ] = m_free_loops[ dim_t::K ].end() - 1;
  }

  //find possible BR kernel dimension
  l_loop = m_free_loops[ dim_t::K ].end() - 2;
  if( m_free_loops[ dim_t::K ].size() > 1 ){
    l_potential_kernel_loop[ PRIM_BR ] = l_loop;
    l_potential_kernel_size[ PRIM_BR ] = l_loop->size;
  }


  //adapt all kernel target sizes depending on what is possible e.g. small k kernel -> choose bigger m target
  int64_t l_kernel_targets[] = {1, m_target_m, m_target_n, m_target_k};
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
  int64_t l_kernel_sizes[] = {1, 1, 1, 1};
  l_kernel_sizes[PRIM_K] = splitLoop( l_potential_kernel_loop[PRIM_K],
                                      &m_free_loops[dim_t::K],
                                      m_loops,
                                      l_kernel_targets[PRIM_K],
                                      0,
                                      exec_t::PRIM );
  l_kernel_sizes[PRIM_N] = splitLoop( l_potential_kernel_loop[PRIM_N],
                                      &m_free_loops[dim_t::N],
                                      m_loops,
                                      l_kernel_targets[PRIM_N],
                                      0,
                                      exec_t::PRIM );
  l_kernel_sizes[PRIM_M] = splitLoop( l_potential_kernel_loop[PRIM_M],
                                      &m_free_loops[dim_t::M],
                                      m_loops,
                                      l_kernel_targets[PRIM_M],
                                      0,
                                      exec_t::PRIM );
  if( l_kernel_targets[PRIM_BR] > 1 ){
      *m_ktype_main = kernel_main_t::BR_MADD;
      l_kernel_sizes[PRIM_BR] = splitLoop( l_potential_kernel_loop[PRIM_BR],
                                           &m_free_loops[dim_t::K],
                                           m_loops,
                                           l_kernel_targets[PRIM_BR],
                                           0,
                                           exec_t::PRIM );
    }
  m_size_all_m /= l_kernel_sizes[PRIM_M];
  m_size_all_n /= l_kernel_sizes[PRIM_N];
}

void einsum_ir::backend::ContractionOptimizer::reorderLoops(){

  //determine parallel targets similarly to kernel targets
  int64_t l_target_parallel_m = sqrt(m_target_parallel);
  int64_t l_target_parallel_n = sqrt(m_target_parallel);
  int64_t l_target_parallel_c = 1;

  while( m_size_all_m <= l_target_parallel_m / 2 ){
    l_target_parallel_m /= 2;
    l_target_parallel_n *= 2;
  }
  while( m_size_all_n <= l_target_parallel_n / 2 ){
    l_target_parallel_n /= 2;
    l_target_parallel_m *= 2;
  }
  while( m_size_all_m <= l_target_parallel_m / 2 ){
    l_target_parallel_m /= 2;
    l_target_parallel_c *= 2;
  }

  //add parallel dimension
  int64_t l_size_parallel = 1;
  l_size_parallel *= add_loops_until(&m_free_loops[dim_t::N],
                                     m_loops, 
                                     l_target_parallel_n,
                                     exec_t::SFC);
  l_size_parallel *= add_loops_until(&m_free_loops[dim_t::M],
                                     m_loops, 
                                     l_target_parallel_m,
                                     exec_t::SFC);
  l_size_parallel *= add_loops_until(&m_free_loops[dim_t::C],
                                     m_loops, 
                                     l_target_parallel_c,
                                     exec_t::OMP);


  //add remaining dimensions
  add_all_loops( &m_free_loops[dim_t::K],
                 m_loops,
                 exec_t::SEQ );
  add_all_loops( &m_free_loops[dim_t::M],
                 m_loops,
                 exec_t::SEQ );
  add_all_loops( &m_free_loops[dim_t::N],
                 m_loops,
                 exec_t::SEQ );
  add_all_loops( &m_free_loops[dim_t::C], 
                 m_loops,
                 exec_t::SEQ );
}

int64_t einsum_ir::backend::ContractionOptimizer::add_loops_until( std::vector<loop_property> * i_source_loops,
                                                                   std::vector<loop_property> * i_dest_loops,
                                                                   int64_t                      i_target_size,
                                                                   exec_t                       i_new_exec_t ){
  int64_t l_size_all = 1;
  int64_t l_target_size = i_target_size;
  while( i_source_loops->size() > 0 && 
         l_target_size > 1 ){
    std::vector<loop_property>::iterator l_it = i_source_loops->end() - 1;
    int64_t l_size_loop  = l_it->size;
    int64_t l_size_split = splitLoop( l_it,
                                      i_source_loops,
                                      i_dest_loops,
                                      l_target_size,
                                      0,
                                      i_new_exec_t );

    l_size_all *= l_size_loop;
    l_target_size = i_target_size / l_size_all;
    if( l_size_split < l_size_loop ){
      l_target_size = 0;
    }
  }

  return l_size_all;
}

int64_t einsum_ir::backend::ContractionOptimizer::add_all_loops( std::vector<loop_property> * i_source_loops,
                                                                 std::vector<loop_property> * i_dest_loops,
                                                                 exec_t                       i_new_exec_t ){
  int64_t l_size_all = 1;
  while( i_source_loops->size() > 0 ){
    std::vector<loop_property>::iterator l_it = i_source_loops->end() - 1;
    l_size_all *= addLoop( l_it,
                           i_source_loops,
                           i_dest_loops,
                           0,
                           i_new_exec_t );
  }
  return l_size_all;
}

int64_t einsum_ir::backend::ContractionOptimizer::splitLoop( std::vector<loop_property>::iterator i_loop,
                                                             std::vector<loop_property> *         i_source_loops,
                                                             std::vector<loop_property> *         i_dest_loops,
                                                             int64_t                              i_target_size,
                                                             int64_t                              i_new_loop_pos, 
                                                             exec_t                               i_new_exec_t ){

  int64_t l_split = findSplit(i_loop->size, i_target_size );

  i_dest_loops->insert(i_dest_loops->begin() + i_new_loop_pos, *i_loop );
  i_dest_loops->at(i_new_loop_pos).exec_type = i_new_exec_t;
  i_dest_loops->at(0).size = l_split;

  i_loop->size           /= l_split;
  i_loop->stride_left    *= l_split;
  i_loop->stride_right   *= l_split;
  i_loop->stride_out_aux *= l_split;
  i_loop->stride_out     *= l_split;

  if( i_loop->size == 1 ){
      i_source_loops->erase( i_loop );
  }
  return l_split;                                                         
}

void einsum_ir::backend::ContractionOptimizer::push_empty_loop( std::vector<loop_property> * i_dest_loops,
                                                                dim_t                        i_new_dim_t ){

  loop_property l_empty_loop;

  l_empty_loop.dim_type = i_new_dim_t;
  l_empty_loop.exec_type = exec_t::SEQ;
  l_empty_loop.size = 1;
  l_empty_loop.stride_left = 0;
  l_empty_loop.stride_right = 0;
  l_empty_loop.stride_out = 0;
  l_empty_loop.stride_out_aux = 0;

  i_dest_loops->push_back( l_empty_loop );                                                       
}

int64_t einsum_ir::backend::ContractionOptimizer::addLoop( std::vector<loop_property>::iterator i_loop,
                                                           std::vector<loop_property> *         i_source_loops,
                                                           std::vector<loop_property> *         i_dest_loops,
                                                           int64_t i_new_loop_pos, 
                                                           exec_t  i_new_exec_t ){
  
  i_dest_loops->insert(i_dest_loops->begin() + i_new_loop_pos, *i_loop );
  i_dest_loops->at(i_new_loop_pos).exec_type = i_new_exec_t;
  i_source_loops->erase( i_loop );
    
  return  i_loop->size;

  return 1;                                                         
}

//TODO optimize. This does not find good kernels bigger than 1024
int64_t einsum_ir::backend::ContractionOptimizer::findSplit( int64_t i_dim_size,
                                                             int64_t i_target_size
                                                           ){
  //factorization of number
  int64_t l_best_factor = i_dim_size;
  double l_best_distance = abs(log((double)i_dim_size/i_target_size));

  int64_t l_max_factor = 1024;
  l_max_factor = i_dim_size / 2 < l_max_factor ? i_dim_size / 2 : l_max_factor;
  for(int64_t l_i = 2; l_i <= l_max_factor; l_i++){
    if(i_dim_size % l_i == 0){
      double l_distance_i = abs(log((double)l_i/i_target_size));
      if(l_best_distance > l_distance_i){
        l_best_factor = l_i;
        l_best_distance = l_distance_i;
      }
    }
  }
  
  return l_best_factor;
}