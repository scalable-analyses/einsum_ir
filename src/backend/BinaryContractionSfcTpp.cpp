#include "BinaryContractionSfcTpp.h"
#include "binaryContraction/ContractionOptimizer.h"

#include <iostream>

einsum_ir::err_t einsum_ir::backend::BinaryContractionSfcTpp::compile() {
  BinaryContraction::compile_base();
  err_t l_err = err_t::UNDEFINED_ERROR;

  // derive strides
  std::map< int64_t, int64_t > l_strides_left;
  std::map< int64_t, int64_t > l_strides_right;
  std::map< int64_t, int64_t > l_strides_out;
  std::map< int64_t, int64_t > l_strides_out_aux;

  strides( m_num_dims_left,
           m_dim_ids_left,
           m_dim_sizes_outer_left,
           &l_strides_left );

  strides( m_num_dims_right,
           m_dim_ids_right,
           m_dim_sizes_outer_right,
           &l_strides_right );

  strides( m_num_dims_out,
           m_dim_ids_out,
           m_dim_sizes_outer_out,
           &l_strides_out );

  if( m_dim_sizes_outer_out_aux != nullptr ) {
    strides( m_num_dims_out,
             m_dim_ids_out,
             m_dim_sizes_outer_out_aux,
             &l_strides_out_aux );
  }

  //get all dimension ids
  std::vector<int64_t> l_all_dim_ids; 
  l_all_dim_ids.reserve( m_dim_ids_c.size() + m_dim_ids_m.size() + m_dim_ids_n.size() + m_dim_ids_k.size() );
  l_all_dim_ids.insert(l_all_dim_ids.end(), m_dim_ids_c.begin(), m_dim_ids_c.end());
  l_all_dim_ids.insert(l_all_dim_ids.end(), m_dim_ids_m.begin(), m_dim_ids_m.end());
  l_all_dim_ids.insert(l_all_dim_ids.end(), m_dim_ids_n.begin(), m_dim_ids_n.end());
  l_all_dim_ids.insert(l_all_dim_ids.end(), m_dim_ids_k.begin(), m_dim_ids_k.end());


  //lower to ContractionOptimizer data structure
  std::vector<loop_property> l_loops;
  l_loops.resize(l_all_dim_ids.size());

  for(size_t l_id = 0; l_id < l_all_dim_ids.size(); l_id++){
    int64_t l_dim_id = l_all_dim_ids[l_id];
    l_loops[l_id].dim_type       = m_dim_types[      l_dim_id];
    l_loops[l_id].exec_type      = einsum_ir::SEQ;
    l_loops[l_id].size           = m_dim_sizes_inner->at(l_dim_id);
    l_loops[l_id].stride_left    = map_find_default<int64_t>(&l_strides_left,    l_dim_id, 0);
    l_loops[l_id].stride_right   = map_find_default<int64_t>(&l_strides_right,   l_dim_id, 0);
    l_loops[l_id].stride_out_aux = map_find_default<int64_t>(&l_strides_out_aux, l_dim_id, 0);
    l_loops[l_id].stride_out     = map_find_default<int64_t>(&l_strides_out,     l_dim_id, 0);
  }

  /*
  std::cout << "before" << std::endl;
  for(size_t l_id = 0; l_id < l_loops.size(); l_id++){
    std::cout << "dtype: "<< l_loops[l_id].dim_type 
              << " etype: " << l_loops[l_id].exec_type 
              << " size: " << l_loops[l_id].size
              << "\tstride_l: " << l_loops[l_id].stride_left
              << "\tstride_r: " << l_loops[l_id].stride_right
              << "\tstride_o: " << l_loops[l_id].stride_out
              << std::endl;
  }
  */

  //optimize loops
  ContractionOptimizer l_optim;

  l_optim.init(&l_loops,
               &m_ktype_main);
  l_optim.optimize();

  /*
  std::cout << "after" << std::endl;
  for(size_t l_id = 0; l_id < l_loops.size(); l_id++){
    std::cout << "dtype: "<< l_loops[l_id].dim_type 
              << " etype: " << l_loops[l_id].exec_type 
              << " size: " << l_loops[l_id].size
              << "\tstride_l: " << l_loops[l_id].stride_left
              << "\tstride_r: " << l_loops[l_id].stride_right
              << "\tstride_o: " << l_loops[l_id].stride_out
              << std::endl;
  }
  */
  
  //compile backend
  m_backend.init( l_loops,
                  m_dtype_left,
                  m_dtype_right,
                  m_dtype_comp,
                  m_dtype_out,
                  m_ktype_first_touch,
                  m_ktype_main,
                  m_ktype_last_touch);
  
  l_err = m_backend.compile();
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  return err_t::SUCCESS;
}

void einsum_ir::backend::BinaryContractionSfcTpp::threading( int64_t i_num_tasks_target  ){

}

void einsum_ir::backend::BinaryContractionSfcTpp::contract( void const * i_tensor_left,
                   void const * i_tensor_right,
                   void       * io_tensor_out ){
  contract( i_tensor_left,
            i_tensor_right,
            nullptr,
            io_tensor_out );
}

void einsum_ir::backend::BinaryContractionSfcTpp::contract( void const * i_tensor_left,
                   void const * i_tensor_right,
                   void const * i_tensor_out_aux,
                   void       * io_tensor_out ){
  m_backend.contract( i_tensor_left,
                      i_tensor_right,
                      i_tensor_out_aux,
                      io_tensor_out );
}

