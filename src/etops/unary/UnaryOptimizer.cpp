#include "UnaryOptimizer.h"
#include <algorithm>

void einsum_ir::etops::UnaryOptimizer::init( std::vector< iter_property > * i_iter_space,
                                             int64_t                        i_num_threads ){
  m_iter_space = i_iter_space;
  m_num_threads = i_num_threads;
}

einsum_ir::etops::err_t einsum_ir::etops::UnaryOptimizer::optimize(){

  if( m_iter_space->size() == 0 ){
    return err_t::COMPILATION_FAILED;
  }

  //optimize unary packing kernel
  std::sort(m_iter_space->begin(), m_iter_space->end(),
            [](iter_property const & a, iter_property const & b) {
              return std::min((double)a.stride_left, a.stride_out + 0.1) > std::min((double)b.stride_left, b.stride_out + 0.1);
            });

  bool l_found_stride_one_in  = false;
  bool l_found_stride_one_out = false;
  std::vector<iter_property>::iterator l_iter;

  //set first loop to primitive
  l_iter = m_iter_space->end() - 1;
  l_iter->exec_type = exec_t::PRIM;
  if( l_iter->stride_left == 1 ){
    l_found_stride_one_in = true;
  }
  if( l_iter->stride_out == 1 ){
    l_found_stride_one_out = true;
  }
  
  //set second loop to primitive
  size_t l_size = m_iter_space->size();
  if(    l_size > 1 
      && (m_iter_space->at(l_size - 2).stride_left == 1 || l_found_stride_one_in)
      && (m_iter_space->at(l_size - 2).stride_out  == 1 || l_found_stride_one_out) ){
    l_iter = m_iter_space->end() - 2;
    l_iter->exec_type = exec_t::PRIM;
  }
  //create extra loop if no two primitive loops found
  else{
    iter_property l_new_iter;
    l_new_iter.exec_type = exec_t::PRIM;
    l_new_iter.size = 1;
    l_new_iter.stride_left = l_iter->size * l_found_stride_one_in  + !l_found_stride_one_in;
    l_new_iter.stride_out =  l_iter->size * l_found_stride_one_out + !l_found_stride_one_out;
    m_iter_space->insert(m_iter_space->end() - l_found_stride_one_in, l_new_iter);
  }

  return err_t::SUCCESS;
}
