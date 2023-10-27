#include "Unary.h"

void einsum_ir::backend::Unary::strides( int64_t                              i_num_dims,
                                         std::map< int64_t, int64_t > const * i_dim_sizes,
                                         int64_t                      const * i_dim_ids_in,
                                         int64_t                      const * i_dim_ids_out,
                                         int64_t                            * o_strides_in,
                                         int64_t                            * o_strides_out ) {
  // strides of the output tensor
  int64_t l_stride_tmp = 1;
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    o_strides_out[ i_num_dims - l_di - 1 ] = l_stride_tmp;

    int64_t l_id = i_dim_ids_out[ i_num_dims - l_di - 1 ];
    l_stride_tmp *= i_dim_sizes->at( l_id );
  }

  // assemble strides of the input tensor
  std::map< int64_t, int64_t > l_strides_in;
  l_stride_tmp = 1;
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    int64_t l_id = i_dim_ids_in[ i_num_dims - l_di - 1 ];
    l_strides_in.insert(  std::pair< int64_t, int64_t >( l_id, l_stride_tmp ) );

    l_stride_tmp *= i_dim_sizes->at( l_id );
  }

  // assign strides of the input tensor based on the order of the output tensor
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    int64_t l_id = i_dim_ids_out[l_di];
    o_strides_in[ l_di ] = l_strides_in.at( l_id );
  }
}

void einsum_ir::backend::Unary::init( int64_t                              i_num_dims,
                                      std::map< int64_t, int64_t > const * i_dim_sizes,
                                      int64_t                      const * i_dim_ids_in,
                                      int64_t                      const * i_dim_ids_out,
                                      data_t                               i_dtype_in,
                                      data_t                               i_dtype_comp,
                                      data_t                               i_dtype_out,
                                      kernel_t                             i_ktype_main ) {
  m_num_dims    = i_num_dims;
  m_dim_sizes   = i_dim_sizes;
  m_dim_ids_in  = i_dim_ids_in;
  m_dim_ids_out = i_dim_ids_out;
  m_dtype_in    = i_dtype_in;
  m_dtype_comp  = i_dtype_comp;
  m_dtype_out   = i_dtype_out;
  m_ktype_main  = i_ktype_main;
}

einsum_ir::err_t einsum_ir::backend::Unary::compile_base() {
  m_sizes_out.clear();
  m_sizes_out.resize( m_num_dims );
  for( int64_t l_di = 0; l_di < m_num_dims; l_di++ ) {
    int64_t l_id = m_dim_ids_out[l_di];
    m_sizes_out[l_di] = m_dim_sizes->at( l_id );
  }

  m_strides_in.clear();
  m_strides_out.clear();

  m_strides_in.resize( m_num_dims );
  m_strides_out.resize( m_num_dims );

  strides( m_num_dims,
           m_dim_sizes,
           m_dim_ids_in,
           m_dim_ids_out,
           m_strides_in.data(),
           m_strides_out.data() );

  return err_t::SUCCESS;
}