#include <ATen/ATen.h>
#include "catch.hpp"
#include "ResidualBlock.h"

TEST_CASE( "Tests the generation of a residual block.", "[residual_block]" ) {
  // allocate data
  int64_t l_width  = 96;
  int64_t l_height = 32;
  int64_t l_num_features = 256;
  
  at::Tensor l_data_activation_0 = at::zeros( { l_num_features, l_height+2, l_width+2 } );
  at::Tensor l_data_weights_0    = at::randn( { l_num_features, l_num_features, 3, 3 } ) / l_num_features;
  at::Tensor l_data_bias_0       = at::randn( { l_num_features, 1, 1 } ) / l_num_features;

  at::Tensor l_data_activation_1 = at::zeros( { l_num_features, l_height+2, l_width+2 } );
  at::Tensor l_data_weights_1    = at::randn( { l_num_features, l_num_features, 3, 3 } ) / l_num_features;
  at::Tensor l_data_bias_1       = at::randn( { l_num_features, 1, 1 } ) / l_num_features;

  // assign input data
  at::Tensor l_data_activation_0_no_pad = l_data_activation_0.narrow(        1, 1, l_height );
  l_data_activation_0_no_pad            = l_data_activation_0_no_pad.narrow( 2, 1, l_width );
  l_data_activation_0_no_pad.copy_( at::randn( {l_num_features, l_height, l_width} ) );

  // permute activation and biases to desired format
  at::Tensor l_data_activation_0_perm = l_data_activation_0.permute( {1, 2, 0} ).clone().contiguous();
  at::Tensor l_data_bias_0_perm       = l_data_bias_0.permute(       {1, 2, 0} ).contiguous();
  at::Tensor l_data_activation_1_perm = l_data_activation_1.permute( {1, 2, 0} ).contiguous();
  at::Tensor l_data_bias_1_perm       = l_data_bias_1.permute(       {1, 2, 0} ).contiguous();

  einsum_ir::frontend::ResidualBlock l_res_block;

  l_res_block.init( l_width,
                    l_height,
                    3,
                    3,
                    l_num_features,
                    1,
                    l_data_activation_0_perm.data_ptr(),
                    l_data_weights_0.data_ptr(),
                    l_data_bias_0_perm.data_ptr(),
                    l_data_activation_1_perm.data_ptr(),
                    l_data_weights_1.data_ptr(),
                    l_data_bias_1_perm.data_ptr(),
                    l_data_activation_0_perm.data_ptr() );

  einsum_ir::err_t l_err = l_res_block.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  // eval
  l_res_block.eval();

  // compute reference
  at::Tensor l_act_1 = at::conv2d( l_data_activation_0,
                                   l_data_weights_0 );
  l_act_1 += l_data_bias_0;
  l_act_1 = at::relu( l_act_1 );

  l_data_activation_0_no_pad += at::conv2d( l_act_1,
                                            l_data_weights_1,
                                            {},
                                            1,
                                            1 ); 
  l_data_activation_0_no_pad += l_data_bias_1;

  at::Tensor l_data_activation_0_perm_no_pad = l_data_activation_0_perm.narrow( 0, 1, l_height );
  l_data_activation_0_perm_no_pad = l_data_activation_0_perm_no_pad.narrow( 1, 1, l_width );

  REQUIRE( at::allclose( l_data_activation_0_perm_no_pad.permute( {2, 0, 1} ),
                         l_data_activation_0_no_pad,
                         1E-5,
                         1E-6 ) );
}

TEST_CASE( "Tests the generation of a residual blocks with strided convolutions", "[residual_block]" ) {
  // allocate data
  int64_t l_width  = 96;
  int64_t l_height = 32;
  int64_t l_num_features = 256;
  int64_t l_stride = 2;
  
  at::Tensor l_data_activation_0 = at::zeros( { l_num_features, l_height+2, l_width+2 } );
  at::Tensor l_data_weights_0    = at::randn( { l_num_features, l_num_features, 3, 3 } ) / l_num_features;
  at::Tensor l_data_bias_0       = at::randn( { l_num_features, 1, 1 } ) / l_num_features;

  at::Tensor l_data_activation_1 = at::zeros( { l_num_features, l_height/l_stride+2, l_width/l_stride+2 } );
  at::Tensor l_data_weights_1    = at::randn( { l_num_features, l_num_features, 3, 3 } ) / l_num_features;
  at::Tensor l_data_bias_1       = at::randn( { l_num_features, 1, 1 } ) / l_num_features;

  at::Tensor l_data_activation_2 = at::zeros( {  l_num_features, l_height/l_stride+2, l_width/l_stride+2 } );

  // assign input data
  at::Tensor l_data_activation_0_no_pad = l_data_activation_0.narrow(        1, 1, l_height );
  l_data_activation_0_no_pad            = l_data_activation_0_no_pad.narrow( 2, 1, l_width );
  l_data_activation_0_no_pad.copy_( at::randn( {l_num_features, l_height, l_width} ) );

  at::Tensor l_data_activation_2_no_pad = l_data_activation_2.narrow(        1, 1, l_height / l_stride );
  l_data_activation_2_no_pad            = l_data_activation_2_no_pad.narrow( 2, 1, l_width / l_stride );
  l_data_activation_2_no_pad.copy_( at::randn( {l_num_features, l_height / l_stride, l_width / l_stride} ) );


  // permute activation and biases to desired format
  at::Tensor l_data_activation_0_perm = l_data_activation_0.permute( {1, 2, 0} ).clone().contiguous();
  at::Tensor l_data_bias_0_perm       = l_data_bias_0.permute(       {1, 2, 0} ).contiguous();
  at::Tensor l_data_activation_1_perm = l_data_activation_1.permute( {1, 2, 0} ).contiguous();
  at::Tensor l_data_bias_1_perm       = l_data_bias_1.permute(       {1, 2, 0} ).contiguous();
  at::Tensor l_data_activation_2_perm = l_data_activation_2.permute( {1, 2, 0} ).clone().contiguous();

  einsum_ir::frontend::ResidualBlock l_res_block;

  l_res_block.init( l_width,
                    l_height,
                    3,
                    3,
                    l_num_features,
                    2,
                    l_data_activation_0_perm.data_ptr(),
                    l_data_weights_0.data_ptr(),
                    l_data_bias_0_perm.data_ptr(),
                    l_data_activation_1_perm.data_ptr(),
                    l_data_weights_1.data_ptr(),
                    l_data_bias_1_perm.data_ptr(),
                    l_data_activation_2_perm.data_ptr() );

  einsum_ir::err_t l_err = l_res_block.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  // eval
  l_res_block.eval();

  // compute reference
  at::Tensor l_act_1 = at::conv2d( l_data_activation_0,
                                   l_data_weights_0,
                                   {},
                                   l_stride );
  l_act_1 += l_data_bias_0;
  l_act_1 = at::relu( l_act_1 );

  l_data_activation_2_no_pad += at::conv2d( l_act_1,
                                            l_data_weights_1,
                                            {},
                                            1,
                                            1 ); 
  l_data_activation_2_no_pad += l_data_bias_1;

  at::Tensor l_data_activation_2_perm_no_pad = l_data_activation_2_perm.narrow( 0, 1, l_height/l_stride );
  l_data_activation_2_perm_no_pad = l_data_activation_2_perm_no_pad.narrow( 1, 1, l_width/l_stride );

  REQUIRE( at::allclose( l_data_activation_2_perm_no_pad.permute( {2, 0, 1} ),
                         l_data_activation_2_no_pad,
                         1E-5,
                         1E-6 ) );
}