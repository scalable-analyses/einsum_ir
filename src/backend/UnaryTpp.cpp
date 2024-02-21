#include "UnaryTpp.h"

libxsmm_datatype einsum_ir::backend::UnaryTpp::dtype_to_libxsmm( data_t i_dtype ) {
  if( i_dtype == FP32 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F32;
  }
  else if( i_dtype == FP64 ) {
    return libxsmm_datatype::LIBXSMM_DATATYPE_F64;
  }

  return libxsmm_datatype::LIBXSMM_DATATYPE_UNSUPPORTED;
}

einsum_ir::err_t einsum_ir::backend::UnaryTpp::compile() {
  err_t l_err = Unary::compile_base();
  if( l_err != einsum_ir::SUCCESS ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  int64_t l_m      = 1;
  int64_t l_n      = 1;
  int64_t l_ld_in  = 1;
  int64_t l_ld_out = 1;

  // identify matrix dimensions
  int64_t l_id_in  = m_dim_ids_in[  m_num_dims - 1 ];
  int64_t l_id_out = m_dim_ids_out[ m_num_dims - 1 ];
  if( m_num_dims > 0 ) {
    l_m = m_dim_sizes->at( l_id_in );

    if( l_id_in != l_id_out ) {
      l_n = m_dim_sizes->at( l_id_out );
    }
  }

  // identify leading dimensions
  for( int64_t l_di = 0; l_di < m_num_dims; l_di++ ) {
    // note: strides are given based dimension ordering of output tensor
    if( m_dim_ids_out[l_di] == l_id_out ) {
      l_ld_in = m_strides_in[l_di];
      break;
    }
  }
  for( int64_t l_di = 0; l_di < m_num_dims; l_di++ ) {
    if( m_dim_ids_out[l_di] == l_id_in ) {
      l_ld_out = m_strides_out[l_di];
      break;
    }
  }

  // assemble loop sizes and strides
  m_loop_sizes.clear();
  m_loop_sizes.reserve( m_num_dims );
  m_loop_strides_in.clear();
  m_loop_strides_out.clear();
  m_loop_strides_in.reserve( m_num_dims );
  m_loop_strides_out.reserve( m_num_dims );

  for( int64_t l_di = 0; l_di < m_num_dims; l_di++ ) {
    if(    m_dim_ids_out[l_di] != l_id_in
        && m_dim_ids_out[l_di] != l_id_out ) {
      m_loop_sizes.push_back( m_sizes_out[l_di] );

      m_loop_strides_in.push_back( m_strides_in[l_di] );
      m_loop_strides_out.push_back( m_strides_out[l_di] );
    }
  }

  // adjust leading dimensions for single row/column cases to satisfy jitter
  l_ld_in  = (l_n == 1) ? l_m : l_ld_in;
  l_ld_out = (l_m == 1) ? l_n : l_ld_out;

  libxsmm_datatype l_xmm_dtype_in   = dtype_to_libxsmm( m_dtype_in );
  libxsmm_datatype l_xmm_dtype_comp = dtype_to_libxsmm( m_dtype_comp );
  libxsmm_datatype l_xmm_dtype_out  = dtype_to_libxsmm( m_dtype_out );

  libxsmm_bitfield l_flag_out_unary = LIBXSMM_MELTW_FLAG_UNARY_NONE;

  libxsmm_meltw_unary_shape l_shape_main = libxsmm_create_meltw_unary_shape( l_m,
                                                                             l_n,
                                                                             l_ld_in,
                                                                             l_ld_out,
                                                                             l_xmm_dtype_in,
                                                                             l_xmm_dtype_out,
                                                                             l_xmm_dtype_comp );

  if( m_ktype_main == COPY ) {
    m_xmm_kernel_main = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT,
                                                      l_shape_main,
                                                      l_flag_out_unary );
  }
  else if( m_ktype_main != UNDEFINED_KTYPE ) {
    return err_t::COMPILATION_FAILED;
  }

  if( m_xmm_kernel_main == nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // init and compile loop interface
  int64_t l_num_loops = m_loop_sizes.size();

  // enforce a single kernel invocation if there are no loops
  if( l_num_loops == 0 ) {
    l_num_loops = 1;
    m_loop_sizes.push_back( 1 );
  }

  m_unary_loops.init( l_num_loops,
                      m_loop_sizes.data(),
                      m_loop_strides_in.data(),
                      m_loop_strides_out.data(),
                      ce_n_bytes( m_dtype_in ),
                      ce_n_bytes( m_dtype_out ),
                      m_xmm_kernel_main );

  l_err = m_unary_loops.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  m_compiled = true;

  return err_t::SUCCESS;
}

void einsum_ir::backend::UnaryTpp::threading( int64_t i_num_tasks_target  ) {
  m_unary_loops.threading( i_num_tasks_target );
}

void einsum_ir::backend::UnaryTpp::eval( void const * i_tensor_in,
                                         void       * io_tensor_out ) {
  m_unary_loops.eval( i_tensor_in,
                      io_tensor_out );
}