#include "ResidualBlock.h"

void einsum_ir::frontend::ResidualBlock::init( int64_t   i_width,
                                               int64_t   i_height,
                                               int64_t   i_kernel_width,
                                               int64_t   i_kernel_height,
                                               int64_t   i_num_features_in,
                                               int64_t   i_num_features_out,
                                               int64_t   i_stride,
                                               void    * i_activations_0,
                                               void    * i_weights_0,
                                               void    * i_bias_0,
                                               void    * io_activations_1,
                                               void    * i_weights_1,
                                               void    * i_bias_1,
                                               void    * io_activations_2 ) {
  m_width            = i_width;
  m_height           = i_height;
  m_kernel_width     = i_kernel_width;
  m_kernel_height    = i_kernel_height;
  m_num_features_in  = i_num_features_in;
  m_num_features_out = i_num_features_out;
  m_stride           = i_stride;
  m_data_abx         = i_activations_0;
  m_data_yxcd        = i_weights_0;
  m_data_aby_aux     = i_bias_0;
  m_data_aby         = io_activations_1;
  m_data_zyef        = i_weights_1;
  m_data_abz_aux     = i_bias_1;
  m_data_abz         = io_activations_2;
}

einsum_ir::err_t einsum_ir::frontend::ResidualBlock::compile() {
  // split input feature dimension until target is reached
  int64_t l_num_features_in = m_num_features_in;
  while( l_num_features_in > m_num_target_features ) {
    if( l_num_features_in % 2 != 0 ) {
      return err_t::COMPILATION_FAILED;
    }
    m_num_features_split_in.push_back( 2 );
    l_num_features_in /= 2;
  }
  m_num_features_split_in.push_back( l_num_features_in );
  int64_t l_num_split_in = m_num_features_split_in.size();

  // split output feature dimension until target is reached
  int64_t l_num_features_out = m_num_features_out;
  while( l_num_features_out > m_num_target_features ) {
    if( l_num_features_out % 2 != 0 ) {
      return err_t::COMPILATION_FAILED;
    }
    m_num_features_split_out.push_back( 2 );
    l_num_features_out /= 2;
  }
  m_num_features_split_out.push_back( l_num_features_out );
  int64_t l_num_split_out = m_num_features_split_out.size();

  // derive padding sizes
  if( (m_kernel_height - 1) % 2 != 0 ) {
    return err_t::COMPILATION_FAILED;
  }
  m_pad_height = (m_kernel_height - 1);

  if( (m_kernel_width - 1) % 2 != 0 ) {
    return err_t::COMPILATION_FAILED;
  }
  m_pad_width = (m_kernel_width - 1);

  if( m_height % m_stride != 0 ) {
    return err_t::COMPILATION_FAILED;
  }
  if( m_width % m_stride != 0 ) {
    return err_t::COMPILATION_FAILED;
  }

  // set inner dimension sizes related to the image and kernel sizes
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0, m_height / m_stride ) );
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1, m_width  / m_stride ) );
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2, m_kernel_height     ) );
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3, m_kernel_width      ) );
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 4, m_kernel_height     ) );
  m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 5, m_kernel_width      ) );
  
  // set inner dimension sizes related to the features
  int64_t l_dim_id = 6;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( l_dim_id, m_num_features_split_in[l_sp] ) );
    l_dim_id++;
  }
  for( int64_t l_char = 0; l_char < 2; l_char++ ) { // yz
    for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
      m_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( l_dim_id, m_num_features_split_out[l_sp] ) );
      l_dim_id++;
    }
  }

  // set outer input dimension sizes related to the input image and kernel sizes
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 0, m_height + m_pad_height ) );
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 1, m_width  + m_pad_width  ) );
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 2, m_kernel_height         ) );
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 3, m_kernel_width          ) );
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 4, m_kernel_height         ) );
  m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 5, m_kernel_width          ) );
  
  // set outer input dimension sizes related to the features
  l_dim_id = 6;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( l_dim_id, m_num_features_split_in[l_sp] ) );
    l_dim_id++;
  }
  for( int64_t l_char = 0; l_char < 2; l_char++ ) { // yz
    for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
      m_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( l_dim_id, m_num_features_split_out[l_sp] ) );
      l_dim_id++;
    }
  }

  // set outer dimension sizes related to the intermediate and output image and kernel sizes
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, m_height / m_stride + m_pad_height ) );
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, m_width  / m_stride + m_pad_width  ) );
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2, m_kernel_height                    ) );
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3, m_kernel_width                     ) );
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 4, m_kernel_height                    ) );
  m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 5, m_kernel_width                     ) );
  
  // set outer dimension sizes related to the intermediate and output features
  l_dim_id = 6;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( l_dim_id, m_num_features_split_in[l_sp] ) );
    l_dim_id++;
  }
  for( int64_t l_char = 0; l_char < 2; l_char++ ) { // yz
    for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
      m_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( l_dim_id, m_num_features_split_out[l_sp] ) );
      l_dim_id++;
    }
  }

  // set aux dimension sizes
  m_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 0, 1 ) );
  m_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 1, 1 ) );
  l_dim_id = 6 + l_num_split_in;
  for( int64_t l_char = 0; l_char < 2; l_char++ ) { // yz
    for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
      m_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( l_dim_id, m_num_features_split_out[l_sp] ) );
      l_dim_id++;
    }
  }

  // set dimension link
  m_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 2, 0 ) );
  m_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 1 ) );
  m_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 4, 0 ) );
  m_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 5, 1 ) );

  // set abx dimension ids
  m_dim_ids_abx.push_back( 0 );
  m_dim_ids_abx.push_back( 1 );
  l_dim_id = 6;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_ids_abx.push_back( l_dim_id );
    l_dim_id++;
  }

  // set yxcd dimension ids
  l_dim_id = 6 + l_num_split_in;
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_ids_yxcd.push_back( l_dim_id );
    l_dim_id++;
  }
  l_dim_id = 6;
  for( int64_t l_sp = 0; l_sp < l_num_split_in; l_sp++ ) {
    m_dim_ids_yxcd.push_back( l_dim_id );
    l_dim_id++;
  }
  m_dim_ids_yxcd.push_back( 2 );
  m_dim_ids_yxcd.push_back( 3 );

  // set aby dimension ids
  m_dim_ids_aby.push_back( 0 );
  m_dim_ids_aby.push_back( 1 );
  l_dim_id = 6 + l_num_split_in;
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_ids_aby.push_back( l_dim_id );
    l_dim_id++;
  }

  // set zyef dimension ids
  l_dim_id = 6 + l_num_split_in + l_num_split_out;
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_ids_zyef.push_back( l_dim_id );
    l_dim_id++;
  }
  l_dim_id = 6 + l_num_split_in;
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_ids_zyef.push_back( l_dim_id );
    l_dim_id++;
  }
  m_dim_ids_zyef.push_back( 2 );
  m_dim_ids_zyef.push_back( 3 );

  // set abz dimension ids
  m_dim_ids_abz.push_back( 0 );
  m_dim_ids_abz.push_back( 1 );
  l_dim_id = 6 + l_num_split_in + l_num_split_out;
  for( int64_t l_sp = 0; l_sp < l_num_split_out; l_sp++ ) {
    m_dim_ids_abz.push_back( l_dim_id );
    l_dim_id++;
  }

  // set offsets
  m_offsets.insert( std::pair< int64_t, int64_t >( 0, 1 ) );
  m_offsets.insert( std::pair< int64_t, int64_t >( 1, 1 ) );

  // set strides
  m_strides.insert( std::pair< int64_t, int64_t >( 0, m_stride ) );
  m_strides.insert( std::pair< int64_t, int64_t >( 1, m_stride ) );

  // init leaf nodes
  m_node_abx.init( 2 + l_num_split_in,
                   m_dim_ids_abx.data(),
                   &m_dim_sizes_inner,
                   &m_dim_sizes_outer_input,
                   einsum_ir::FP32,
                   m_data_abx );

  m_node_yxcd.init( 2 + l_num_split_in+l_num_split_out,
                    m_dim_ids_yxcd.data(),
                    &m_dim_sizes_inner,
                    &m_dim_sizes_outer,
                    einsum_ir::FP32,
                    m_data_yxcd );

  m_node_zyef.init( 2 + 2*l_num_split_out,
                    m_dim_ids_zyef.data(),
                    &m_dim_sizes_inner,
                    &m_dim_sizes_outer,
                    einsum_ir::FP32,
                    m_data_zyef );

  // dependent nodes
  m_node_aby.init( 2 + l_num_split_out,
                   m_dim_ids_aby.data(),
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
                   m_data_aby_aux,
                   m_data_aby,
                   einsum_ir::kernel_t::COPY,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::RELU,
                   &m_node_abx,
                   &m_node_yxcd );

  m_node_abz.init( 2 + l_num_split_out,
                   m_dim_ids_abz.data(),
                   &m_dim_sizes_inner,
                   &m_dim_sizes_aux_outer,
                   &m_dim_sizes_outer,
                   nullptr,
                   &m_offsets,
                   nullptr,
                   nullptr,
                   nullptr,
                   &m_dim_link_s_to_p,
                   einsum_ir::FP32,
                   m_data_abz_aux,
                   m_data_abz,
                   einsum_ir::kernel_t::ADD,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &m_node_aby,
                   &m_node_zyef );

  // compile
  einsum_ir::err_t l_err = m_node_abz.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return l_err;
  }

  // store the weights internally in the desired data layout
  m_node_yxcd.store_and_lock_data();
  m_node_zyef.store_and_lock_data();

  return err_t::SUCCESS;
}

void einsum_ir::frontend::ResidualBlock::eval() {
  m_node_abz.eval();
}