#include "UnaryTpp.h"
#include "../basic/unary/UnaryOptimizer.h"

einsum_ir::err_t einsum_ir::backend::UnaryTpp::compile() {
  err_t l_err = Unary::compile_base();
  if( l_err != einsum_ir::SUCCESS ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  //lower to UnaryOptimizer data structure
  std::vector<basic::iter_property> l_loops;
  l_loops.resize(m_num_dims);
  for(std::size_t l_di = 0; l_di < l_loops.size(); l_di++){
    l_loops[l_di].exec_type   = basic::exec_t::SEQ;
    l_loops[l_di].size        = m_sizes_out[l_di];
    l_loops[l_di].stride_left = m_strides_in[l_di];
    l_loops[l_di].stride_out  = m_strides_out[l_di];
  }

  //convert kernel to basic
  basic::kernel_t l_ktype_main = ce_kernelt_to_basic(m_ktype_main);

  //convert dtype
  basic::data_t l_dtype_in   = ce_dtype_to_basic(m_dtype_in);
  basic::data_t l_dtype_comp = ce_dtype_to_basic(m_dtype_comp);
  basic::data_t l_dtype_out  = ce_dtype_to_basic(m_dtype_out);

  //optimize loops
  einsum_ir::basic::UnaryOptimizer l_optim;

  l_optim.init( &l_loops ,
                m_num_threads,
                false );
  l_optim.optimize();

  //setup backend
  m_backend.init( l_loops,
                  l_dtype_in,
                  l_dtype_comp,
                  l_dtype_out,
                  l_ktype_main,
                  m_num_threads );

  l_err = ce_basic_err_to_err(m_backend.compile());
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  return err_t::SUCCESS;
}

void einsum_ir::backend::UnaryTpp::eval( void const * i_tensor_in,
                                         void       * io_tensor_out ) {
  m_backend.eval( i_tensor_in, io_tensor_out );
}