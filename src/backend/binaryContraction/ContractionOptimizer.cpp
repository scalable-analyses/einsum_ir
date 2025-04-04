#include "ContractionOptimizer.h"
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::backend::ContractionOptimizer::init( std::vector< loop_property > * i_loops,
                                                     kernel_t                     * i_ktype_main ){
  m_loops = i_loops;
  m_ktype_main = i_ktype_main;
}

void einsum_ir::backend::ContractionOptimizer::optimize(){
  //sort loops depeding on strides, exec_types and dim_types
  sortLoops();

  // fuse loops if possible
  fuseLoops();

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

void einsum_ir::backend::ContractionOptimizer::reorderLoops(){
  //add unoptimized loops to internal data structures
  std::vector< std::vector< loop_property > > l_free_loops;
  std::vector< loop_property >                l_reordered_loops;

  int64_t l_possible_parallelism = 1;
  l_free_loops.resize(4);
  std::vector<loop_property>::iterator l_it = m_loops->begin();
  while( l_it < m_loops->end() ){
    if( l_it->exec_type == einsum_ir::SEQ ){
      if( l_it->dim_type == einsum_ir::M || l_it->dim_type == einsum_ir::N ){
        l_possible_parallelism *= l_it->size;
      }
      l_free_loops[l_it->dim_type].push_back(*l_it);
      m_loops->erase(l_it);
    }
    else{
      l_it++;
    }
  }
  //TODO should determine the stage of optimization to support partially optimized input

  //find possible kernel dimensions
  std::vector<loop_property>::iterator l_potential_kernel_loop[] = { l_free_loops[ einsum_ir::K ].end(),
                                                                     l_free_loops[ einsum_ir::M ].end(),
                                                                     l_free_loops[ einsum_ir::N ].end(),
                                                                     l_free_loops[ einsum_ir::K ].end() };  
  int64_t l_potential_kernel_size[] = {1,1,1,1};
  if( l_free_loops[ einsum_ir::M ].size() > 0 ){
    loop_property l_loop = l_free_loops[ einsum_ir::M ].back();
    if( l_loop.stride_left == 1 && 
        l_loop.stride_out  == 1 ){
      l_potential_kernel_loop[1]--;
      l_potential_kernel_size[1] *= l_potential_kernel_loop[1]->size;
    }
  }
  if( l_free_loops[ einsum_ir::N ].size() > 0 ){
    l_potential_kernel_loop[2]--;
    l_potential_kernel_size[2] *= l_potential_kernel_loop[2]->size;
  }
  if( l_free_loops[ einsum_ir::K ].size() > 0 ){
    loop_property l_loop = l_free_loops[ einsum_ir::K ].back();
    if( l_loop.stride_right == 1 ){
      l_potential_kernel_loop[3]--;
      l_potential_kernel_size[3] *= l_potential_kernel_loop[3]->size;
      if( l_free_loops[ einsum_ir::K ].size() > 1 ){
        l_potential_kernel_loop[0] -= 2;
        l_potential_kernel_size[0] *= l_potential_kernel_loop[0]->size;
      }
    }
    else{
      l_potential_kernel_loop[0]--;
      l_potential_kernel_size[0] *= l_potential_kernel_loop[0]->size;
    }     
  }


  //adapt all kernel target sizes depending on what is possible e.g. small k kernel -> choose bigger m target
  int64_t l_kernel_targets[] = {1, m_target_m, m_target_n, m_target_k};
  while( l_potential_kernel_size[3] <= l_kernel_targets[3] / 2 ){
    l_kernel_targets[3] /= 2;
    l_kernel_targets[0] *= 2;
  }
  while( l_potential_kernel_size[0] <= l_kernel_targets[0] / 2 ){
    l_kernel_targets[0] /= 2;
    l_kernel_targets[1] *= 2;
  }
  while( l_potential_kernel_size[1] <= l_kernel_targets[1] / 2 ){
    l_kernel_targets[1] /= 2;
    l_kernel_targets[2] *= 2;
  }

  //reduce kernel size when parallelism is low
  while(l_possible_parallelism / (l_kernel_targets[1] * l_kernel_targets[2]) < m_num_tasks &&
        l_kernel_targets[1] * l_kernel_targets[2] > 1 ){
    if(l_kernel_targets[1] < l_kernel_targets[2]){
      l_kernel_targets[2] /= 2;
    }
    else{
      l_kernel_targets[1] /= 2;
    }
  }
 
  //create kernel from potential kernel dimensions
  int64_t l_kernel_sizes[] = {1, 1, 1, 1};
  l_kernel_sizes[3] = splitLoop( l_potential_kernel_loop[3],
                                 &l_free_loops[einsum_ir::K],
                                 m_loops,
                                 l_kernel_targets[3],
                                 0,
                                 einsum_ir::PRIM );
  l_kernel_sizes[2] = splitLoop( l_potential_kernel_loop[2],
                                 &l_free_loops[einsum_ir::N],
                                 m_loops,
                                 l_kernel_targets[2],
                                 0,
                                 einsum_ir::PRIM );
  l_kernel_sizes[1] = splitLoop( l_potential_kernel_loop[1],
                                 &l_free_loops[einsum_ir::M],
                                 m_loops,\
                                 l_kernel_targets[1],
                                 0,
                                 einsum_ir::PRIM );
  //TODO remove false when seg fault is fixed
  if( l_kernel_targets[0] > 1 ){
    *m_ktype_main = einsum_ir::BR_MADD;
    l_kernel_sizes[0] = splitLoop( l_potential_kernel_loop[0],
                                  &l_free_loops[einsum_ir::K],
                                  m_loops,
                                  l_kernel_targets[0],
                                  0,
                                  einsum_ir::PRIM );
  }

  //add sfc n dimension
  int64_t l_size_sfc = 1;
  l_size_sfc *= add_loops_until(&l_free_loops[einsum_ir::N],
                                m_loops, 
                                sqrt(m_target_parallel),
                                einsum_ir::SFC);

  //add sfc m dimension
  l_size_sfc *= add_loops_until(&l_free_loops[einsum_ir::M],
                                m_loops, 
                                sqrt(m_target_parallel),
                                einsum_ir::SFC);



  //TODO: add omp dimensions
  int64_t l_target_parallel = m_target_parallel / l_size_sfc;

  //add remaining dimensions
  add_all_loops( &l_free_loops[einsum_ir::K],
                 m_loops,
                 einsum_ir::SEQ );

  add_all_loops( &l_free_loops[einsum_ir::M],
                 m_loops,
                 einsum_ir::SEQ );

  add_all_loops( &l_free_loops[einsum_ir::N],
                 m_loops,
                 einsum_ir::SEQ );

  add_all_loops( &l_free_loops[einsum_ir::C], 
                  m_loops,
                  einsum_ir::SEQ );
}

//TODO should have a bug for split last loop (while loop not ending)
int64_t einsum_ir::backend::ContractionOptimizer::add_loops_until( std::vector<loop_property> * i_source_loops,
                                                                   std::vector<loop_property> * i_dest_loops,
                                                                   int64_t i_target_size,
                                                                   exec_t  i_new_exec_t ){
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
                                                                 exec_t  i_new_exec_t ){
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
                                                             int64_t i_target_size,
                                                             int64_t i_new_loop_pos, 
                                                             exec_t  i_new_exec_t ){


  if( i_source_loops->end() > i_loop ){  
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
  
  std::cout << "WARNING: did not find loop, adding size 1 loop"<< std::endl;
  //add empty loops for empty dimesions
  loop_property l_empty_loop;
  l_empty_loop.dim_type = dim_t::UNDEFINED_DIM;
  l_empty_loop.exec_type = exec_t::PRIM;
  l_empty_loop.size = 1;
  l_empty_loop.stride_left = 0;
  l_empty_loop.stride_right = 0;
  l_empty_loop.stride_out = 0;
  l_empty_loop.stride_out_aux = 0;

  i_dest_loops->insert(i_dest_loops->begin() + i_new_loop_pos, l_empty_loop );
  return 1;                                                         
}

int64_t einsum_ir::backend::ContractionOptimizer::addLoop( std::vector<loop_property>::iterator i_loop,
                                                           std::vector<loop_property> *         i_source_loops,
                                                           std::vector<loop_property> *         i_dest_loops,
                                                           int64_t i_new_loop_pos, 
                                                           exec_t  i_new_exec_t ){
  if( i_source_loops->end() >= i_loop ){  
    i_dest_loops->insert(i_dest_loops->begin() + i_new_loop_pos, *i_loop );
    i_dest_loops->at(i_new_loop_pos).exec_type = i_new_exec_t;
    i_source_loops->erase( i_loop );
    
    return  i_loop->size;
  }

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