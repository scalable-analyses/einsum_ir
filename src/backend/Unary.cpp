#include "Unary.h"

void einsum_ir::backend::Unary::strides( int64_t                              i_num_dims,
                                         std::map< int64_t, int64_t > const * i_dim_sizes,
                                         int64_t                      const * i_dim_ids,
                                         int64_t                            * o_strides) {
int64_t l_stride_tmp = 1;
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    o_strides[ i_num_dims - l_di - 1 ] = l_stride_tmp;

    int64_t l_id = i_dim_ids[ i_num_dims - l_di - 1 ];
    l_stride_tmp *= i_dim_sizes->at( l_id );
  }
}

void einsum_ir::backend::Unary::order_strides_output_based( int64_t         i_num_dims,
                                                            int64_t const * i_dim_ids_in,
                                                            int64_t const * i_dim_ids_out,
                                                            int64_t       * io_strides) {
  std::map< int64_t, int64_t > l_strides_in;
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    int64_t l_id = i_dim_ids_in[ l_di ];
    l_strides_in.insert(  std::pair< int64_t, int64_t >( l_id, io_strides[ l_di ] ) );
  }

  // assign strides of the input tensor based on the order of the output tensor
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    int64_t l_id = i_dim_ids_out[l_di];
    io_strides[ l_di ] = l_strides_in.at( l_id );
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

void einsum_ir::backend::Unary::init( int64_t                              i_num_dims,
                                      std::map< int64_t, int64_t > const * i_dim_sizes,
                                      int64_t                      const * i_dim_ids_in,
                                      int64_t                      const * i_dim_ids_out,
                                      int64_t                            * i_strides_in,
                                      int64_t                            * i_strides_out,
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

  if(i_strides_in != nullptr){
    m_strides_in  = std::vector<int64_t>(i_strides_in, i_strides_in + i_num_dims);
  }
  if(i_strides_out != nullptr){
    m_strides_out = std::vector<int64_t>(i_strides_out, i_strides_out + i_num_dims);
  }
}

einsum_ir::err_t einsum_ir::backend::Unary::compile_base() {
  m_sizes_out.clear();
  m_sizes_out.resize( m_num_dims );
  for( int64_t l_di = 0; l_di < m_num_dims; l_di++ ) {
    int64_t l_id = m_dim_ids_out[l_di];
    m_sizes_out[l_di] = m_dim_sizes->at( l_id );
  }

  if(m_strides_in.empty()){
    m_strides_in.clear();
    m_strides_in.resize( m_num_dims );

    strides( m_num_dims,
             m_dim_sizes,
             m_dim_ids_in,
             m_strides_in.data() );
  }
  if(m_strides_out.empty()){
    m_strides_out.clear();
    m_strides_out.resize( m_num_dims );

    strides( m_num_dims,
             m_dim_sizes,
             m_dim_ids_out,
             m_strides_out.data() );
  }

  order_strides_output_based( m_num_dims,
                              m_dim_ids_in,
                              m_dim_ids_out,
                              m_strides_in.data() );

  return err_t::SUCCESS;
}