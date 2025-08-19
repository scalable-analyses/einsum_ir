#include "BinaryContractionScalar.h"
#include "../basic/binary/ContractionOptimizer.h"

einsum_ir::err_t einsum_ir::backend::BinaryContractionScalar::compile() {
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
  else if(    m_ktype_first_touch == kernel_t::ADD
           || m_ktype_first_touch == kernel_t::COPY ) { 
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
  std::vector<basic::iter_property> l_loops;
  l_loops.resize(l_all_dim_ids.size());

  for(std::size_t l_id = 0; l_id < l_all_dim_ids.size(); l_id++){
    int64_t l_dim_id = l_all_dim_ids[l_id];
    l_loops[l_id].dim_type       = ce_dimt_to_basic(m_dim_types[l_dim_id]);
    l_loops[l_id].exec_type      = basic::exec_t::SEQ;
    l_loops[l_id].size           = m_dim_sizes_inner->at(l_dim_id);
    l_loops[l_id].stride_left    = map_find_default<int64_t>(&l_strides_left,    l_dim_id, 0);
    l_loops[l_id].stride_right   = map_find_default<int64_t>(&l_strides_right,   l_dim_id, 0);
    l_loops[l_id].stride_out_aux = map_find_default<int64_t>(&l_strides_out_aux, l_dim_id, 0);
    l_loops[l_id].stride_out     = map_find_default<int64_t>(&l_strides_out,     l_dim_id, 0);
  }

  //convert kernel to basic
  basic::kernel_t l_ktype_first_touch = ce_kernelt_to_basic(m_ktype_first_touch);
  basic::kernel_t l_ktype_main        = ce_kernelt_to_basic(m_ktype_main);
  basic::kernel_t l_ktype_last_touch  = ce_kernelt_to_basic(m_ktype_last_touch);

  //convert dtype
  basic::data_t l_dtype_left  = ce_dtype_to_basic(m_dtype_left);
  basic::data_t l_dtype_right = ce_dtype_to_basic(m_dtype_right);
  basic::data_t l_dtype_comp  = ce_dtype_to_basic(m_dtype_comp);
  basic::data_t l_dtype_out   = ce_dtype_to_basic(m_dtype_out);

  //optimize loops
  einsum_ir::basic::ContractionOptimizer l_optim;

  int64_t l_num_threads_m = 1;
  int64_t l_num_threads_n = 1;
  l_optim.init(&l_loops,
               &l_ktype_main,
               m_num_threads,
               m_target_prim_m,
               m_target_prim_n,
               m_target_prim_k,
               false,
               basic::packed_gemm_t::ALL_STRIDE_ONE,
               ce_n_bytes(m_dtype_out),
               m_l2_cache_size,
               &l_num_threads_m,
               &l_num_threads_n );
  l_optim.optimize();

  einsum_ir::basic::ContractionMemoryManager * l_contraction_memory = nullptr;
  if( m_memory != nullptr ){
    l_contraction_memory = m_memory->get_contraction_memory_manager();
  }
  
  //compile backend
  m_backend.init( l_loops,
                  l_dtype_left,
                  l_dtype_right,
                  l_dtype_comp,
                  l_dtype_out,
                  l_ktype_first_touch,
                  l_ktype_main,
                  l_ktype_last_touch,
                  l_num_threads_m,
                  l_num_threads_n,
                  l_contraction_memory );
  
  l_err = ce_basic_err_to_err(m_backend.compile());
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  return err_t::SUCCESS;
}

void einsum_ir::backend::BinaryContractionScalar::contract( void const * i_tensor_left,
                                                            void const * i_tensor_right,
                                                            void const * i_tensor_out_aux,
                                                            void       * io_tensor_out ) {
  m_backend.contract( i_tensor_left,
                      i_tensor_right,
                      i_tensor_out_aux,
                      io_tensor_out );
}

void einsum_ir::backend::BinaryContractionScalar::contract( void const * i_tensor_left,
                                                            void const * i_tensor_right,
                                                            void       * io_tensor_out ) {
  contract( i_tensor_left,
            i_tensor_right,
            nullptr,
            io_tensor_out );
}