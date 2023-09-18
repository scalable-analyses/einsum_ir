#include "ResidualBlockDownSample.h"

void einsum_ir::frontend::ResidualBlockDownSample::init( int64_t   i_width,
                                                         int64_t   i_height,
                                                         int64_t   i_kernel_width,
                                                         int64_t   i_kernel_height,
                                                         int64_t   i_kernel_width_down_sample,
                                                         int64_t   i_kernel_height_down_sample,
                                                         int64_t   i_num_features_in,
                                                         int64_t   i_num_features_out,
                                                         int64_t   i_stride,
                                                         void    * i_activations_0,
                                                         void    * i_weights_down_sample,
                                                         void    * i_bias_down_sample,
                                                         void    * i_weights_0,
                                                         void    * i_bias_0,
                                                         void    * io_activations_1,
                                                         void    * i_weights_1,
                                                         void    * i_bias_1,
                                                         void    * io_activations_2 ) {
  m_width                     = i_width;
  m_height                    = i_height;
  m_kernel_width              = i_kernel_width;
  m_kernel_height             = i_kernel_height;
  m_kernel_width_down_sample  = i_kernel_width_down_sample;
  m_kernel_height_down_sample = i_kernel_height_down_sample;
  m_num_features_in           = i_num_features_in;
  m_num_features_out          = i_num_features_out;
  m_stride                    = i_stride;

  m_data_abx                  = i_activations_0;
  m_data_abu                  = io_activations_2;
  m_data_abu_aux              = i_bias_down_sample;
  m_data_uxcd                 = i_weights_down_sample;

  m_res_block.init( i_width,
                    i_height,
                    i_kernel_width,
                    i_kernel_height,
                    i_num_features_in,
                    i_num_features_out,
                    i_stride,
                    i_activations_0,
                    i_weights_0,
                    i_bias_0,
                    io_activations_1,
                    i_weights_1,
                    i_bias_1,
                    io_activations_2 );
}

einsum_ir::err_t einsum_ir::frontend::ResidualBlockDownSample::compile() {
  /*
   * compile the base block
   */
  err_t l_err = m_res_block.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return l_err;
  }

  /* 
   * compile the additional convolution
   */
  std::vector< int64_t > const & l_num_features_split_in = m_res_block.m_num_features_split_in;
  int64_t l_num_split_in = l_num_features_split_in.size();

  std::vector< int64_t > const & l_num_features_split_out = m_res_block.m_num_features_split_out;
  int64_t l_num_split_out = l_num_features_split_out.size();

  // check that we have either equal kernels or 1 in downsampling
  if(    m_kernel_height != m_kernel_height_down_sample
      || m_kernel_width  != m_kernel_width_down_sample ) {
    if(    m_kernel_width_down_sample != 1
        || m_kernel_height_down_sample != 1 ) {
      return err_t::COMPILATION_FAILED;
    }
  }

  int64_t l_pad_width = m_res_block.m_pad_width;
  int64_t l_pad_height = m_res_block.m_pad_height;

  // set inner dimension sizes related to the image and kernel sizes
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0, m_height / m_stride         ) );
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1, m_width  / m_stride         ) );
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2, m_kernel_height_down_sample ) );
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3, m_kernel_width_down_sample  ) );
  int64_t l_dim_id = 4;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( l_dim_id, l_num_features_split_in[l_sp] ) );
    l_dim_id++;
  }
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( l_dim_id, l_num_features_split_out[l_sp] ) );
    l_dim_id++;
  }

  // set outer input dimension sizes related to the input image and kernel sizes
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 0, m_height + l_pad_height     ) );
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 1, m_width  + l_pad_width      ) );
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 2, m_kernel_height_down_sample ) );
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 3, m_kernel_width_down_sample  ) );
  
  // set outer input dimension sizes related to the features
  l_dim_id = 4;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( l_dim_id, l_num_features_split_in[l_sp] ) );
    l_dim_id++;
  }
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( l_dim_id, l_num_features_split_out[l_sp] ) );
    l_dim_id++;
  }

  // set outer dimension sizes related to the output image and kernel sizes
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, m_height / m_stride + l_pad_height ) );
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, m_width  / m_stride + l_pad_width  ) );
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2, m_kernel_height_down_sample        ) );
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3, m_kernel_width_down_sample         ) );

  // set outer dimension sizes related to the output features
  l_dim_id = 4;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( l_dim_id, l_num_features_split_in[l_sp] ) );
    l_dim_id++;
  }
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( l_dim_id, l_num_features_split_out[l_sp] ) );
    l_dim_id++;
  }

  // set aux dimension sizes
  m_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 0, 1 ) );
  m_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 1, 1 ) );
  l_dim_id = 4 + l_num_split_in;
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( l_dim_id, l_num_features_split_out[l_sp] ) );
    l_dim_id++;
  }

  // set dimension link
  m_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 2, 0 ) );
  m_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 1 ) );

  // set abx dimension ids
  m_dim_ids_abx.push_back( 0 );
  m_dim_ids_abx.push_back( 1 );
  l_dim_id = 4;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_ids_abx.push_back( l_dim_id );
    l_dim_id++;
  }

  // set uxcd dimension ids
  l_dim_id = 4 + l_num_split_in;
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_ids_uxcd.push_back( l_dim_id );
    l_dim_id++;
  }
  l_dim_id = 4;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_ids_uxcd.push_back( l_dim_id );
    l_dim_id++;
  }
  m_dim_ids_uxcd.push_back( 2 );
  m_dim_ids_uxcd.push_back( 3 );

  // set abu dimension ids
  m_dim_ids_abu.push_back( 0 );
  m_dim_ids_abu.push_back( 1 );
  l_dim_id = 4 + l_num_split_in;
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_ids_abu.push_back( l_dim_id );
    l_dim_id++;
  }

  // set offsets
  m_offsets.insert( std::pair< int64_t, int64_t >( 0, l_pad_width/2 ) );
  m_offsets.insert( std::pair< int64_t, int64_t >( 1, l_pad_height/2 ) );

  // set strides
  m_strides.insert( std::pair< int64_t, int64_t >( 0, m_stride ) );
  m_strides.insert( std::pair< int64_t, int64_t >( 1, m_stride ) );

  // TODO: implement properly, leads to invalid reads if data is copied to internal data structures
  if(    m_kernel_width_down_sample == 1
      && m_kernel_height_down_sample == 1 ){
    int64_t l_off_abx = 0;
    l_off_abx =  l_pad_height/2 * (m_width + l_pad_width) * m_num_features_in;
    l_off_abx += l_pad_width/2 * m_num_features_in;
    l_off_abx *= sizeof(float);
    m_data_abx = (char *) m_data_abx + l_off_abx;
  }

  // init leaf nodes
  m_node_abx.init( 2 + l_num_split_in,
                   m_dim_ids_abx.data(),
                   &m_dim_sizes_inner,
                   &m_dim_sizes_outer_input,
                   einsum_ir::FP32,
                   m_data_abx );

  m_node_uxcd.init( 2 + l_num_split_in + l_num_split_out,
                    m_dim_ids_uxcd.data(),
                    &m_dim_sizes_inner,
                    &m_dim_sizes_outer,
                    einsum_ir::FP32,
                    m_data_uxcd );

  // dependent node
  m_node_abu.init( 2 + l_num_split_out,
                   m_dim_ids_abu.data(),
                   &m_dim_sizes_inner,
                   &m_dim_sizes_aux_outer,
                   &m_dim_sizes_outer,
                   nullptr,
                   &m_offsets,
                   &m_strides,
                   nullptr,
                   nullptr,
                   &m_dim_link_s_to_p,
                   einsum_ir::FP32,
                   m_data_abu_aux,
                   m_data_abu,
                   einsum_ir::kernel_t::ADD,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &m_node_abx,
                   &m_node_uxcd );

  // compile
  l_err = m_node_abu.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return l_err;
  }

  // abort compilation for offsets and transposes
  if(    m_kernel_width_down_sample == 1
      && m_kernel_height_down_sample == 1 ){
    if( m_node_abx.m_data_ptr_int != nullptr ) {
      return einsum_ir::COMPILATION_FAILED;
    }
  }

  m_node_uxcd.store_and_lock_data();

  return einsum_ir::SUCCESS;
}

void einsum_ir::frontend::ResidualBlockDownSample::eval() {
  m_node_abu.eval();
  m_res_block.eval();
}

int64_t einsum_ir::frontend::ResidualBlockDownSample::num_ops() {
  int64_t l_num_ops  = m_node_abu.num_ops();
  l_num_ops         += m_res_block.num_ops();

  return l_num_ops;
}