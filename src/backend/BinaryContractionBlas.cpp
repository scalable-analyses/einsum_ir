#include "BinaryContractionBlas.h"
#include "../binary/ContractionOptimizer.h"

einsum_ir::err_t einsum_ir::backend::BinaryContractionBlas::compile() {
  err_t l_err = err_t::UNDEFINED_ERROR;

  l_err = BinaryContraction::compile_base();
  if( l_err != einsum_ir::SUCCESS ) {
    return l_err;
  }

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
  else {
    l_strides_out_aux = l_strides_out;
  }

  //get all dimension ids
  std::vector<int64_t> l_all_dim_ids; 
  l_all_dim_ids.reserve( m_dim_ids_c.size() + m_dim_ids_m.size() + m_dim_ids_n.size() + m_dim_ids_k.size() );
  l_all_dim_ids.insert(l_all_dim_ids.end(), m_dim_ids_c.begin(), m_dim_ids_c.end());
  l_all_dim_ids.insert(l_all_dim_ids.end(), m_dim_ids_m.begin(), m_dim_ids_m.end());
  l_all_dim_ids.insert(l_all_dim_ids.end(), m_dim_ids_n.begin(), m_dim_ids_n.end());
  l_all_dim_ids.insert(l_all_dim_ids.end(), m_dim_ids_k.begin(), m_dim_ids_k.end());


  //lower to ContractionOptimizer data structure
  std::vector<binary::iter_property> l_loops;
  l_loops.resize(l_all_dim_ids.size());

  for(size_t l_id = 0; l_id < l_all_dim_ids.size(); l_id++){
    int64_t l_dim_id = l_all_dim_ids[l_id];
    l_loops[l_id].dim_type       = m_dim_types[l_dim_id];
    l_loops[l_id].exec_type      = binary::exec_t::SEQ;
    l_loops[l_id].size           = m_dim_sizes_inner->at(l_dim_id);
    l_loops[l_id].stride_left    = map_find_default<int64_t>(&l_strides_left,    l_dim_id, 0);
    l_loops[l_id].stride_right   = map_find_default<int64_t>(&l_strides_right,   l_dim_id, 0);
    l_loops[l_id].stride_out_aux = map_find_default<int64_t>(&l_strides_out_aux, l_dim_id, 0);
    l_loops[l_id].stride_out     = map_find_default<int64_t>(&l_strides_out,     l_dim_id, 0);
  }

  // check supported kernel types
  if(    m_ktype_first_touch != einsum_ir::kernel_t::UNDEFINED_KTYPE
      && m_ktype_first_touch != einsum_ir::kernel_t::ZERO
      && m_ktype_first_touch != einsum_ir::kernel_t::CPX_ZERO ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }
  if(    m_ktype_main != einsum_ir::kernel_t::MADD
      && m_ktype_main != einsum_ir::kernel_t::CPX_MADD ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }
  if( m_ktype_last_touch != einsum_ir::kernel_t::UNDEFINED_KTYPE ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }

  // set CPX dimension correctly
  if( m_ktype_main == einsum_ir::kernel_t::CPX_MADD ) {
    l_loops[0].dim_type = CPX;
  }

  //optimize loops
  einsum_ir::binary::ContractionOptimizer l_optim;

  l_optim.init(&l_loops,
               &m_ktype_main,
               m_num_threads,
               m_target_prim_m,
               m_target_prim_n,
               m_target_prim_k,
               false,
               true );
  l_optim.optimize();

  //compile backend
  m_backend.init( l_loops,
                  m_dtype_left,
                  m_dtype_right,
                  m_dtype_comp,
                  m_dtype_out,
                  m_ktype_first_touch,
                  m_ktype_main,
                  m_ktype_last_touch,
                  m_num_threads );
  
  l_err = m_backend.compile();
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  return err_t::SUCCESS;
}

void einsum_ir::backend::BinaryContractionBlas::contract( void const * i_tensor_left,
                                                          void const * i_tensor_right,
                                                          void       * io_tensor_out ){
  contract( i_tensor_left,
            i_tensor_right,
            nullptr,
            io_tensor_out );
}

void einsum_ir::backend::BinaryContractionBlas::contract( void const * i_tensor_left,
                                                          void const * i_tensor_right,
                                                          void const * i_tensor_out_aux,
                                                          void       * io_tensor_out ){
  m_backend.contract( i_tensor_left,
                      i_tensor_right,
                      i_tensor_out_aux,
                      io_tensor_out );
}

