#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>
#include "frontend/ResidualBlock.h"
#include "frontend/ResidualBlockDownSample.h"

/**
 * Benchmarks the evaluation of a standard ResNet block using separate ATen operators. 
 *
 * @param i_data_input input data.
 * @param i_conv_weights_0 weight of the first convolution.
 * @param i_conv_biases_0 biases of the first convolution.
 * @param i_conv_weights_1 weight of the second convolution.
 * @param i_conv_biases_1 biases of the second convolution.
 * @param i_conv_weights_2 weight of the third convolution.
 * @param i_conv_biases_2 biases of the third convolution.
 * @param i_conv_weights_3 weight of the fourth convolution.
 * @param i_conv_biases_3 biases of the fourth convolution.
 * @param o_result will be set to the result of the block.
 * @param o_time_eval will be set to the evaluation time of the block.
 **/
void bench_eager( at::Tensor & i_data_input,
                  at::Tensor & i_conv_weights_0,
                  at::Tensor & i_conv_biases_0,
                  at::Tensor & i_conv_weights_1,
                  at::Tensor & i_conv_biases_1,
                  at::Tensor & i_conv_weights_2,
                  at::Tensor & i_conv_biases_2,
                  at::Tensor & i_conv_weights_fused_3,
                  at::Tensor & i_conv_biases_3,
                  at::Tensor & o_result,
                  double     & o_time_eval ) {
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  l_tp0 = std::chrono::steady_clock::now();
  at::Tensor l_tmp_0 = at::conv2d( i_data_input,
                                   i_conv_weights_0,
                                   i_conv_biases_0,
                                   1,
                                   1 );

  l_tmp_0 = at::relu( l_tmp_0 );

  at::Tensor l_tmp_1 = at::conv2d( l_tmp_0,
                                   i_conv_weights_1,
                                   i_conv_biases_1,
                                   1,
                                   1 );

  l_tmp_1 = at::relu( l_tmp_1 + i_data_input );

  at::Tensor l_tmp_2 = at::conv2d( l_tmp_1,
                                   i_conv_weights_2,
                                   i_conv_biases_2,
                                   1,
                                   1 );

  l_tmp_2 = at::relu( l_tmp_2 );

  at::Tensor l_tmp_3 = at::conv2d( l_tmp_2,
                                   i_conv_weights_fused_3,
                                   i_conv_biases_3,
                                   1,
                                   1 );

  o_result = at::relu( l_tmp_3 + l_tmp_1 );
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  o_time_eval = l_dur.count();
}

/**
 * Benchmarks the evaluation of a ResNet block with downsampling using separate ATen operators. 
 *
 * @param i_data_input input data.
 * @param i_conv_weights_0 weight of the first convolution.
 * @param i_conv_biases_0 biases of the first convolution.
 * @param i_conv_weights_1 weight of the second convolution.
 * @param i_conv_biases_1 biases of the second convolution.
 * @param i_conv_weights_2 weight of the third convolution.
 * @param i_conv_biases_2 biases of the third convolution.
 * @param i_conv_weights_3 weight of the fourth convolution.
 * @param i_conv_biases_3 biases of the fourth convolution.
 * @param i_down_sample_conv_weights weights of the convolution executed in the residual downsampling part.
 * @param i_down_sample_conv_biases weights of the convolution executed in the residual downsampling part.
 * @param o_result will be set to the result of the block.
 * @param o_time_eval will be set to the evaluation time of the block.
 **/
void bench_eager_down_sample( at::Tensor & l_data_input,
                              at::Tensor & i_conv_weights_0,
                              at::Tensor & i_conv_biases_0,
                              at::Tensor & i_conv_weights_1,
                              at::Tensor & i_conv_biases_1,
                              at::Tensor & i_conv_weights_2,
                              at::Tensor & i_conv_biases_2,
                              at::Tensor & i_conv_weights_3,
                              at::Tensor & i_conv_biases_3,
                              at::Tensor & i_down_sample_conv_weights,
                              at::Tensor & i_down_sample_conv_biases,
                              at::Tensor & o_result,
                              double     & o_time_eval ) {
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  l_tp0 = std::chrono::steady_clock::now();
  at::Tensor l_tmp_0 = at::conv2d( l_data_input,
                                   i_conv_weights_0,
                                   i_conv_biases_0,
                                   2,
                                   1 );
  l_tmp_0 = at::relu( l_tmp_0 );

  at::Tensor l_tmp_1 = at::conv2d( l_tmp_0,
                                   i_conv_weights_1,
                                   i_conv_biases_1,
                                   1,
                                   1 );

  at::Tensor l_tmp_2 = at::conv2d( l_data_input,
                                   i_down_sample_conv_weights,
                                   i_down_sample_conv_biases,
                                   2 );
  l_tmp_2 = at::relu( l_tmp_1 + l_tmp_2 );

  at::Tensor l_tmp_3 = at::conv2d( l_tmp_2,
                                   i_conv_weights_2,
                                   i_conv_biases_2,
                                   1,
                                   1 );

  l_tmp_3 = at::relu( l_tmp_3 );

  at::Tensor l_tmp_4 = at::conv2d( l_tmp_3,
                                   i_conv_weights_3,
                                   i_conv_biases_3,
                                   1,
                                   1 );

  o_result = at::relu( l_tmp_4 + l_tmp_2 );
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  o_time_eval = l_dur.count();
}

/**
 * Benchmarks the execution of a standard ResNet block using einsum_ir.
 *
 * @param i_width input width.
 * @param i_height input width.
 * @param i_kernel_width kernel width.
 * @param i_kernel_height kernel height.
 * @param i_num_features_in number of input features.
 * @param i_num_features_out number of output and intermediate features.
 * @param i_stride used stride in the first convolution.
 * @param i_warm_up_run if true a warmup run is conducted before the time measuring one.
 * @param i_conv_weights_0 weight of the first convolution.
 * @param i_conv_biases_0 biases of the first convolution.
 * @param i_conv_weights_1 weight of the second convolution.
 * @param i_conv_biases_1 biases of the second convolution.
 * @param i_conv_weights_2 weight of the third convolution.
 * @param i_conv_biases_2 biases of the third convolution.
 * @param i_conv_weights_3 weight of the fourth convolution.
 * @param i_conv_biases_3 biases of the fourth convolution.
 * @param io_data_activiation pointer to the input and output data.
 * @param o_data_activation_1 pointer to tensor for intermediate activations.
 * @param o_num_ops will be set to the number operations required for a single execution of the block.
 * @param o_time_compile will be set to the time spent while compiling the einsum tree.
 * @param o_time_eval will be set to the time spent executing the block (once).
 **/
void bench_einsum_ir( int64_t   i_width,
                      int64_t   i_height,
                      int64_t   i_kernel_width,
                      int64_t   i_kernel_height,
                      int64_t   i_num_features_in,
                      int64_t   i_num_features_out,
                      int64_t   i_stride,
                      bool      i_warm_up_run,
                      void    * i_conv_weights_0,
                      void    * i_conv_biases_0,
                      void    * i_conv_weights_1,
                      void    * i_conv_biases_1,
                      void    * i_conv_weights_2,
                      void    * i_conv_biases_2,
                      void    * i_conv_weights_3,
                      void    * i_conv_biases_3,
                      void    * io_data_activation_0,
                      void    * o_data_activation_1,
                      int64_t & o_num_ops,
                      double  & o_time_compile,
                      double  & o_time_eval ) {
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  // init first residual block
  einsum_ir::frontend::ResidualBlock l_res_block_0;
  l_res_block_0.init( i_width,
                      i_height,
                      i_kernel_width,
                      i_kernel_height,
                      i_num_features_in,
                      i_num_features_out,
                      i_stride,
                      io_data_activation_0,
                      i_conv_weights_0,
                      i_conv_biases_0,
                      o_data_activation_1,
                      i_conv_weights_1,
                      i_conv_biases_1,
                      io_data_activation_0 );

  // compile first residual block
  l_tp0 = std::chrono::steady_clock::now();
  einsum_ir::err_t l_err = l_res_block_0.compile();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  o_time_compile = l_dur.count();
  if( l_err != einsum_ir::err_t::SUCCESS ) {
    std::cerr << "error: failed to compile residual block" << std::endl;
  }

  // init second residual block
  einsum_ir::frontend::ResidualBlock l_res_block_1;
  l_res_block_1.init( i_width,
                      i_height,
                      i_kernel_width,
                      i_kernel_height,
                      i_num_features_in,
                      i_num_features_out,
                      i_stride,
                      io_data_activation_0,
                      i_conv_weights_2,
                      i_conv_biases_2,
                      o_data_activation_1,
                      i_conv_weights_3,
                      i_conv_biases_3,
                      io_data_activation_0 );

  // compile second residual block
  l_tp0 = std::chrono::steady_clock::now();
  l_err = l_res_block_1.compile();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  o_time_compile += l_dur.count();
  if( l_err != einsum_ir::err_t::SUCCESS ) {
    std::cerr << "error: failed to compile residual block" << std::endl;
  }

  o_num_ops = l_res_block_0.num_ops() + l_res_block_1.num_ops();

  if( i_warm_up_run ) {
    l_res_block_0.eval();
    l_res_block_1.eval();
  }

  l_tp0 = std::chrono::steady_clock::now();
  l_res_block_0.eval();
  l_res_block_1.eval();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  o_time_eval = l_dur.count();
}

/**
 * Benchmarks the execution of a ResNet block with downsampling using einsum_ir.
 *
 * @param i_width input width.
 * @param i_height input width.
 * @param i_kernel_width kernel width.
 * @param i_kernel_height kernel height.
 * @param i_kernel_width_down_sample kernel width of the downsampling convolution.
 * @param i_kernel_height_down_sample kernel height of the downsampling convolution.
 * @param i_num_features_in number of input features.
 * @param i_num_features_out number of output and intermediate features.
 * @param i_stride_first_block used stride in the first convolution of the first block (with downsampling).
 * @param i_stride_second_block stride used in the the second block.
 * @param i_warm_up_run if true a warmup run is conducted before the time measuring one.
 * @param i_conv_weights_0 weight of the first convolution.
 * @param i_conv_biases_0 biases of the first convolution.
 * @param i_conv_weights_1 weight of the second convolution.
 * @param i_conv_biases_1 biases of the second convolution.
 * @param i_conv_weights_2 weight of the third convolution.
 * @param i_conv_biases_2 biases of the third convolution.
 * @param i_conv_weights_3 weight of the fourth convolution.
 * @param i_conv_biases_3 biases of the fourth convolution.
 * @param i_down_sample_conv_weights weights of the convolution executed in the residual downsampling part.
 * @param i_down_sample_conv_biases weights of the convolution executed in the residual downsampling part.
 * @param i_data_activiation pointer to the input data.
 * @param o_data_activation_1 pointer to a tensor holding intermediate activations.
 * @param o_data_activation_2 pointer to the output data.
 * @param o_num_ops will be set to the number operations required for a single execution of the block.
 * @param o_time_compile will be set to the time spent while compiling the einsum tree.
 * @param o_time_eval will be set to the time spent executing the block (once).
 **/
void bench_einsum_ir_down_sample( int64_t   i_width,
                                  int64_t   i_height,
                                  int64_t   i_kernel_width,
                                  int64_t   i_kernel_height,
                                  int64_t   i_kernel_width_down_sample,
                                  int64_t   i_kernel_height_down_sample,
                                  int64_t   i_num_features_in,
                                  int64_t   i_num_features_out,
                                  int64_t   i_stride_first_block,
                                  int64_t   i_stride_second_block,
                                  bool      i_warm_up_run,
                                  void    * i_conv_weights_0,
                                  void    * i_conv_biases_0,
                                  void    * i_conv_weights_1,
                                  void    * i_conv_biases_1,
                                  void    * i_conv_weights_2,
                                  void    * i_conv_biases_2,
                                  void    * i_conv_weights_3,
                                  void    * i_conv_biases_3,
                                  void    * i_down_sample_conv_weights,
                                  void    * i_down_sample_conv_biases,
                                  void    * i_data_activation_0,
                                  void    * o_data_activation_1,
                                  void    * o_data_activation_2,
                                  int64_t & o_num_ops,
                                  double  & o_time_compile,
                                  double  & o_time_eval ) {
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  // init first residual block
  einsum_ir::frontend::ResidualBlockDownSample l_res_block_0;
  l_res_block_0.init( i_width,
                      i_height,
                      i_kernel_width,
                      i_kernel_height,
                      i_kernel_width_down_sample,
                      i_kernel_height_down_sample,
                      i_num_features_in,
                      i_num_features_out,
                      i_stride_first_block,
                      i_data_activation_0,
                      i_down_sample_conv_weights,
                      i_down_sample_conv_biases,
                      i_conv_weights_0,
                      i_conv_biases_0,
                      o_data_activation_1,
                      i_conv_weights_1,
                      i_conv_biases_1,
                      o_data_activation_2 );

  // compile first residual block
  l_tp0 = std::chrono::steady_clock::now();
  einsum_ir::err_t l_err = l_res_block_0.compile();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  o_time_compile = l_dur.count();
  if( l_err != einsum_ir::err_t::SUCCESS ) {
    std::cerr << "error: failed to compile residual block" << std::endl;
  }

  // init second residual block
  einsum_ir::frontend::ResidualBlock l_res_block_1;
  l_res_block_1.init( i_width / i_stride_first_block,
                      i_height / i_stride_first_block,
                      i_kernel_width,
                      i_kernel_height,
                      i_num_features_out,
                      i_num_features_out,
                      i_stride_second_block,
                      o_data_activation_2,
                      i_conv_weights_2,
                      i_conv_biases_2,
                      o_data_activation_1,
                      i_conv_weights_3,
                      i_conv_biases_3,
                      o_data_activation_2 );

  // compile second residual block
  l_tp0 = std::chrono::steady_clock::now();
  l_err = l_res_block_1.compile();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  o_time_compile += l_dur.count();
  if( l_err != einsum_ir::err_t::SUCCESS ) {
    std::cerr << "error: failed to compile residual block" << std::endl;
  }

  o_num_ops = l_res_block_0.num_ops() + l_res_block_1.num_ops();

  if( i_warm_up_run ) {
    l_res_block_0.eval();
    l_res_block_1.eval();
  }

  l_tp0 = std::chrono::steady_clock::now();
  l_res_block_0.eval();
  l_res_block_1.eval();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  o_time_eval = l_dur.count();
}

/**
 * Fuses a batch norm into a 2d convolution by modifying the weights and the biases.
 * Source: https://github.com/pytorch/pytorch/blob/70ca3ee951ba3dc58e5e42a96bcca0245d8f12bc/torch/nn/utils/fusion.py#L16
 *
 * @param i_conv_w original weights of the convolution.
 * @param i_conv_b original biases of the convolution.
 * @param i_bn_rm running mean of the batch norm.
 * @param i_bn_rv running var of the batch norm.
 * @param i_bn_w weights of the batch norm.
 * @param i_bn_b biases of the batch norm.
 * @param o_conv_w_fused will be set to modified weights of the convolution.
 * @param o_conv_b_fused will be set to modified biases of the convolution.
 **/

void fuse_conv_bn_weights( at::Tensor const  i_conv_w,
                           at::Tensor const  i_conv_b,
                           at::Tensor const  i_bn_rm,
                           at::Tensor const  i_bn_rv,
                           double            i_bn_eps,
                           at::Tensor const  i_bn_w,
                           at::Tensor const  i_bn_b,
                           at::Tensor       &o_conv_w_fused,
                           at::Tensor       &o_conv_b_fused ) {
  at::Tensor l_bn_var_rsqrt = at::rsqrt( i_bn_rv + i_bn_eps );
  o_conv_w_fused = i_conv_w * ( i_bn_w * l_bn_var_rsqrt ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);
  o_conv_b_fused = (i_conv_b - i_bn_rm) * l_bn_var_rsqrt * i_bn_w + i_bn_b;
}

int main( int     i_argc,
          char  * i_argv[] ) {
  if( i_argc < 2 ) {
    std::cerr << "usage: ./bench_resnet path_to_resnet_layer.pt" << std::endl;
    return EXIT_FAILURE;
  }
  std::string l_model_path( i_argv[1] );

  std::cout << "running bench_resnet using " << l_model_path << " as input" << std::endl;

  // inference guard
  c10::InferenceMode l_guard;
  std::cout << "inference guard is_enabled: " << l_guard.is_enabled() << std::endl;

  // read model
  torch::jit::script::Module l_model;

  // note: training pytorch version has to match
  try {
    l_model = torch::jit::load( l_model_path );
  }
  catch( const c10::Error& l_err ) {
    std::cerr << "error: could not load model" << std::endl;
    std::cerr << "message: " << l_err.msg() << std::endl;
    return EXIT_FAILURE;
  }

  // get parameters and print info about them
  std::cout << "parameters:" << std::endl;

  std::vector< at::Tensor > l_conv_weights;
  std::vector< at::Tensor > l_bn_weights;
  std::vector< at::Tensor > l_bn_biases;

  std::vector< at::Tensor > l_down_sample_conv_weights;
  std::vector< at::Tensor > l_down_sample_bn_weights;
  std::vector< at::Tensor > l_down_sample_bn_biases;

  auto l_named_params = l_model.named_parameters( true );
  for( const auto & l_par: l_named_params ) {
    std::cout << "  name: "        << l_par.name               << std::endl;
    std::cout << "  n_dimension: " << l_par.value.ndimension() << std::endl;
    std::cout << "  sizes: "       << l_par.value.sizes()      << std::endl;
    std::cout << "  type: "        << l_par.value.dtype()      << std::endl;
    std::cout << std::endl;

    // standard parts
    if(    l_par.name.find( "conv" )   != std::string::npos
        && l_par.name.find( "weight" ) != std::string::npos ) {
      l_conv_weights.push_back( l_par.value );
    }
    if(    l_par.name.find( "bn" )     != std::string::npos
        && l_par.name.find( "weight" ) != std::string::npos ) {
      l_bn_weights.push_back( l_par.value );
    }
    if(    l_par.name.find( "bn" )   != std::string::npos
        && l_par.name.find( "bias" ) != std::string::npos ) {
      l_bn_biases.push_back( l_par.value );
    }

    // downsampling-related parts
    if(    l_par.name.find( "downsample.0" )   != std::string::npos
        && l_par.name.find( "weight" ) != std::string::npos ) {
      l_down_sample_conv_weights.push_back( l_par.value );
    }
    if(    l_par.name.find( "downsample.1" )     != std::string::npos
        && l_par.name.find( "weight" ) != std::string::npos ) {
      l_down_sample_bn_weights.push_back( l_par.value );
    }
    if(    l_par.name.find( "downsample.1" )   != std::string::npos
        && l_par.name.find( "bias" ) != std::string::npos ) {
      l_down_sample_bn_biases.push_back( l_par.value );
    }
  }

  std::vector< at::Tensor > l_bn_running_means;
  std::vector< at::Tensor > l_bn_running_vars;
  std::vector< at::Tensor > l_down_sample_bn_running_means;
  std::vector< at::Tensor > l_down_sample_bn_running_vars;

  std::cout << "processing attributes" << std::endl;
  auto l_named_atts = l_model.named_attributes( true );
  for( const auto & l_att: l_named_atts ) {
    std::cout << "  name: " << l_att.name << std::endl;
    // standard parts
    if(    l_att.name.find( "bn" )   != std::string::npos
        && l_att.name.find( "running_mean" ) != std::string::npos ) {
      l_bn_running_means.push_back( l_att.value.toTensor() );
    }
    if(    l_att.name.find( "bn" )   != std::string::npos
        && l_att.name.find( "running_var" ) != std::string::npos ) {
      l_bn_running_vars.push_back( l_att.value.toTensor() );
    }
    // residual downsampling-related parts
    if(    l_att.name.find( "downsample.1" )   != std::string::npos
        && l_att.name.find( "running_mean" ) != std::string::npos ) {
      l_down_sample_bn_running_means.push_back( l_att.value.toTensor() );
    }
    if(    l_att.name.find( "downsample.1" )   != std::string::npos
        && l_att.name.find( "running_var" ) != std::string::npos ) {
      l_down_sample_bn_running_vars.push_back( l_att.value.toTensor() );
    }
  }

  std::cout << "extracted tensors:"                                                        << std::endl;
  std::cout << "  #conv weights:                " << l_conv_weights.size()                 << std::endl;
  std::cout << "  #bn weights:                  " << l_bn_weights.size()                   << std::endl;
  std::cout << "  #bn biases:                   " << l_bn_biases.size()                    << std::endl;
  std::cout << "  #bn running means:            " << l_bn_biases.size()                    << std::endl;
  std::cout << "  #bn running vars:             " << l_bn_biases.size()                    << std::endl;
  std::cout << "  #downsample conv weights:     " << l_down_sample_conv_weights.size()     << std::endl;
  std::cout << "  #downsample bn weights:       " << l_down_sample_bn_weights.size()       << std::endl;
  std::cout << "  #downsample bn biases:        " << l_down_sample_bn_biases.size()        << std::endl;
  std::cout << "  #downsample bn running means: " << l_down_sample_bn_running_means.size() << std::endl;
  std::cout << "  #downsample bn running vars:  " << l_down_sample_bn_running_vars.size()  << std::endl;

  // derive fused conv weights and biases
  std::vector< at::Tensor > l_conv_weights_fused( l_conv_weights.size() );
  std::vector< at::Tensor > l_conv_biases_fused(  l_conv_weights.size() );
  std::vector< at::Tensor > l_down_sample_conv_weights_fused( l_down_sample_conv_weights.size() );
  std::vector< at::Tensor > l_down_sample_conv_biases_fused(  l_down_sample_conv_weights.size() );

  for( std::size_t l_co = 0; l_co < l_conv_weights.size(); l_co++ ) {
    fuse_conv_bn_weights( l_conv_weights[l_co],
                          at::zeros_like( l_bn_biases[l_co] ),
                          l_bn_running_means[l_co],
                          l_bn_running_vars[l_co],
                          1.0E-5,
                          l_bn_weights[l_co],
                          l_bn_biases[l_co],
                          l_conv_weights_fused[l_co],
                          l_conv_biases_fused[l_co] );
  }

  for( std::size_t l_co = 0; l_co < l_down_sample_conv_weights.size(); l_co++ ) {
    fuse_conv_bn_weights( l_down_sample_conv_weights[l_co],
                          at::zeros_like( l_bn_biases[l_co] ),
                          l_down_sample_bn_running_means[l_co],
                          l_down_sample_bn_running_vars[l_co],
                          1.0E-5,
                          l_down_sample_bn_weights[l_co],
                          l_down_sample_bn_biases[l_co],
                          l_down_sample_conv_weights_fused[l_co],
                          l_down_sample_conv_biases_fused[l_co] );
  }

  // check for correct sizes
  if(    l_conv_weights_fused.size() != 4
      || l_conv_biases_fused.size()  != 4 ) {
    std::cerr << "error: failed to parse residual block" << std::endl;
    return EXIT_FAILURE;
  }

  // extract properties of the residual block
  bool l_down_sample = false;
  if(         l_down_sample_conv_weights_fused.size() == 0
           && l_down_sample_conv_biases_fused.size()  == 0 ) {
    l_down_sample = false;
  }
  else if(    l_down_sample_conv_weights_fused.size() == 1
           && l_down_sample_conv_biases_fused.size()  == 1 ) {
    l_down_sample = true;
  }
  else {
    std::cerr << "error: failed to parse residual block" << std::endl;
    return EXIT_FAILURE;
  }

  int64_t l_num_features_in  = l_conv_weights_fused[0].sizes()[1];
  int64_t l_num_features_out = l_conv_weights_fused[0].sizes()[0];

  int64_t l_kernel_width  = l_conv_weights_fused[0].sizes()[3];
  int64_t l_kernel_height = l_conv_weights_fused[0].sizes()[2];

  int64_t l_kernel_width_down_sample  = 0;
  int64_t l_kernel_height_down_sample = 0;
  if( l_down_sample ) {
    l_kernel_width_down_sample  = l_down_sample_conv_weights_fused[0].sizes()[3];
    l_kernel_height_down_sample = l_down_sample_conv_weights_fused[0].sizes()[2];
  }

  int64_t l_pad_width  = l_kernel_width-1;
  int64_t l_pad_height = l_kernel_height-1;

  int64_t l_stride = (l_down_sample == false) ? 1 : 2;

  std::cout << "extracted properties of the residual block:" << std::endl;
  std::cout << "  num_features_in:           " << l_num_features_in << std::endl;
  std::cout << "  num_features_out:          " << l_num_features_out << std::endl;
  std::cout << "  kernel_width:              " << l_kernel_width << std::endl;
  std::cout << "  kernel_height:             " << l_kernel_height << std::endl;
  std::cout << "  kernel_width_down_sample:  " << l_kernel_width_down_sample << std::endl;
  std::cout << "  kernel_height_down_sample: " << l_kernel_height_down_sample << std::endl;
  std::cout << "  stride:                    " << l_stride << std::endl;

  // set image properties
  int64_t l_width  = 56;
  int64_t l_height = 56;

  // set up input data
  at::Tensor l_data_activation_0 = at::zeros( { l_num_features_in,
                                                l_height + l_pad_height,
                                                l_width  + l_pad_width } );

  at::Tensor l_data_activation_1 = at::zeros( { l_num_features_out,
                                                l_height/l_stride + l_pad_height,
                                                l_width/l_stride  + l_pad_width } );

  at::Tensor l_data_activation_2 = at::zeros( { l_num_features_out,
                                                l_height/l_stride + l_pad_height,
                                                l_width/l_stride  + l_pad_width } );

  // assign data
  at::Tensor l_data_activation_0_no_pad = l_data_activation_0.narrow(        1, l_pad_height/2, l_height );
  l_data_activation_0_no_pad            = l_data_activation_0_no_pad.narrow( 2, l_pad_width/2,  l_width );
  l_data_activation_0_no_pad.copy_( at::randn( { l_num_features_in,
                                                 l_height,
                                                 l_width } ) / l_num_features_in );
  l_data_activation_0_no_pad = l_data_activation_0_no_pad.unsqueeze(0).contiguous();

  /*
   * performance data structures
   */
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_num_flops = 0;
  double l_time_compile_einsum_ir = 0;
  double l_time_compile_ts = 0;
  double l_time_eval = 0;
  double l_time_total = 0;
  double l_gflops_eval = 0;
  double l_gflops_total = 0;

  /*
   * reference: torchscript
   */
  l_tp0 = std::chrono::steady_clock::now();
  l_model = torch::jit::optimize_for_inference( l_model );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile_ts = l_dur.count();

  at::Tensor l_out_torch = l_model.forward( { l_data_activation_0_no_pad } ).toTensor();

  /*
   * verification: eager ATen execution
   */
  at::Tensor l_out_eager;
  if( l_down_sample == false ) {
    bench_eager( l_data_activation_0_no_pad,
                l_conv_weights_fused[0],
                l_conv_biases_fused[0],
                l_conv_weights_fused[1],
                l_conv_biases_fused[1],
                l_conv_weights_fused[2],
                l_conv_biases_fused[2],
                l_conv_weights_fused[3],
                l_conv_biases_fused[3],
                l_out_eager,
                l_time_eval );
  }
  else {
    bench_eager_down_sample( l_data_activation_0_no_pad,
                             l_conv_weights_fused[0],
                             l_conv_biases_fused[0],
                             l_conv_weights_fused[1],
                             l_conv_biases_fused[1],
                             l_conv_weights_fused[2],
                             l_conv_biases_fused[2],
                             l_conv_weights_fused[3],
                             l_conv_biases_fused[3],
                             l_down_sample_conv_weights_fused[0],
                             l_down_sample_conv_biases_fused[0],
                             l_out_eager,
                             l_time_eval );
  }

  // compare eager to libtorch
  if( !torch::allclose( l_out_eager,
                        l_out_torch,
                        1E-3,
                        1E-5 ) ) {
    std::cerr << "*************************************************" << std::endl;
    std::cerr << "* warning: eager ATen is not close to libtorch! *" << std::endl;
    std::cerr << "*************************************************" << std::endl;
  }

  /*
   * verification: einsum_ir
   */
  at::Tensor l_data_activation_0_perm = l_data_activation_0.permute( {1, 2, 0} ).clone().contiguous();
  at::Tensor l_data_activation_1_perm = l_data_activation_1.permute( {1, 2, 0} ).contiguous();
  at::Tensor l_data_activation_2_perm = l_data_activation_2.permute( {1, 2, 0} ).contiguous();


  if( l_down_sample == false ) {
    bench_einsum_ir( l_width,
                     l_height,
                     l_kernel_width,
                     l_kernel_height,
                     l_num_features_in,
                     l_num_features_out,
                     l_stride,
                     false,
                     l_conv_weights_fused[0].data_ptr(),
                     l_conv_biases_fused[0].data_ptr(),
                     l_conv_weights_fused[1].data_ptr(),
                     l_conv_biases_fused[1].data_ptr(),
                     l_conv_weights_fused[2].data_ptr(),
                     l_conv_biases_fused[2].data_ptr(),
                     l_conv_weights_fused[3].data_ptr(),
                     l_conv_biases_fused[3].data_ptr(),
                     l_data_activation_0_perm.data_ptr(),
                     l_data_activation_1_perm.data_ptr(),
                     l_num_flops,
                     l_time_compile_einsum_ir,
                     l_time_eval );
  } else {
    bench_einsum_ir_down_sample( l_width,
                                 l_height,
                                 l_kernel_width,
                                 l_kernel_height,
                                 l_kernel_width_down_sample,
                                 l_kernel_height_down_sample,
                                 l_num_features_in,
                                 l_num_features_out,
                                 l_stride,
                                 1,
                                 false,
                                 l_conv_weights_fused[0].data_ptr(),
                                 l_conv_biases_fused[0].data_ptr(),
                                 l_conv_weights_fused[1].data_ptr(),
                                 l_conv_biases_fused[1].data_ptr(),
                                 l_conv_weights_fused[2].data_ptr(),
                                 l_conv_biases_fused[2].data_ptr(),
                                 l_conv_weights_fused[3].data_ptr(),
                                 l_conv_biases_fused[3].data_ptr(),
                                 l_down_sample_conv_weights_fused[0].data_ptr(),
                                 l_down_sample_conv_biases_fused[0].data_ptr(),
                                 l_data_activation_0_perm.data_ptr(),
                                 l_data_activation_1_perm.data_ptr(),
                                 l_data_activation_2_perm.data_ptr(),
                                 l_num_flops,
                                 l_time_compile_einsum_ir,
                                 l_time_eval );
  }

  at::Tensor l_data_activation_0_perm_no_pad = l_data_activation_0_perm.narrow( 0, 1, l_height/l_stride );
  at::Tensor l_result_einsum_ir_no_pad;
  if( l_down_sample == false ) {
    l_result_einsum_ir_no_pad = l_data_activation_0_perm.narrow(  0, 1, l_height );
    l_result_einsum_ir_no_pad = l_result_einsum_ir_no_pad.narrow( 1, 1, l_width );
  }
  else {
    l_result_einsum_ir_no_pad = l_data_activation_2_perm.narrow(  0, 1, l_height/l_stride );
    l_result_einsum_ir_no_pad = l_result_einsum_ir_no_pad.narrow( 1, 1, l_width/l_stride );
  }

  if( !torch::allclose( l_result_einsum_ir_no_pad.permute( { 2, 0, 1 } ),
                        l_out_torch.squeeze(),
                        1E-3,
                        1E-5 ) ) {
    std::cerr << "*********************************************" << std::endl;
    std::cerr << "* warning: einsum_ir is not close libtorch! *" << std::endl;
    std::cerr << "*********************************************" << std::endl;
  }

  /*
   * benchmark einsum_ir
   */
  std::cout << "benchmarking einsum_ir" << std::endl;

  double l_dummy_time = 0;
  if( l_down_sample == false ) {
    bench_einsum_ir( l_width,
                    l_height,
                    l_kernel_width,
                    l_kernel_height,
                    l_num_features_in,
                    l_num_features_out,
                    l_stride,
                    true,
                    l_conv_weights_fused[0].data_ptr(),
                    l_conv_biases_fused[0].data_ptr(),
                    l_conv_weights_fused[1].data_ptr(),
                    l_conv_biases_fused[1].data_ptr(),
                    l_conv_weights_fused[2].data_ptr(),
                    l_conv_biases_fused[2].data_ptr(),
                    l_conv_weights_fused[3].data_ptr(),
                    l_conv_biases_fused[3].data_ptr(),
                    l_data_activation_0_perm.data_ptr(),
                    l_data_activation_1_perm.data_ptr(),
                    l_num_flops,
                    l_dummy_time, // dummy here to include JITting overhead
                    l_time_eval );
  }
  else {
    bench_einsum_ir_down_sample( l_width,
                                 l_height,
                                 l_kernel_width,
                                 l_kernel_height,
                                 l_kernel_width_down_sample,
                                 l_kernel_height_down_sample,
                                 l_num_features_in,
                                 l_num_features_out,
                                 l_stride,
                                 1,
                                 true,
                                 l_conv_weights_fused[0].data_ptr(),
                                 l_conv_biases_fused[0].data_ptr(),
                                 l_conv_weights_fused[1].data_ptr(),
                                 l_conv_biases_fused[1].data_ptr(),
                                 l_conv_weights_fused[2].data_ptr(),
                                 l_conv_biases_fused[2].data_ptr(),
                                 l_conv_weights_fused[3].data_ptr(),
                                 l_conv_biases_fused[3].data_ptr(),
                                 l_down_sample_conv_weights_fused[0].data_ptr(),
                                 l_down_sample_conv_biases_fused[0].data_ptr(),
                                 l_data_activation_0_perm.data_ptr(),
                                 l_data_activation_1_perm.data_ptr(),
                                 l_data_activation_2_perm.data_ptr(),
                                 l_num_flops,
                                 l_dummy_time,
                                 l_time_eval );
  }

  l_gflops_eval = 1.0E-9 * l_num_flops / l_time_eval;
  l_time_total = l_time_compile_einsum_ir + l_time_eval;
  l_gflops_total = 1.0E-9 * l_num_flops / (l_time_total);

  std::cout << "  #flops:         " << l_num_flops << std::endl;\
  std::cout << "  time (compile): " << l_time_compile_einsum_ir << std::endl;
  std::cout << "  time (eval):    " << l_time_eval << std::endl;
  std::cout << "  gflops (eval):  " << l_gflops_eval << std::endl;
  std::cout << "  gflops (total): " << l_gflops_total << std::endl;

  /*
   * benchmark torchscript model
   */
  std::cout << "benchmarking libtorch" << std::endl;

  // warmup run
  l_model.forward( { l_data_activation_0_no_pad } );

  l_tp0 = std::chrono::steady_clock::now();
  l_model.forward( { l_data_activation_0_no_pad } );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_eval = l_dur.count();

  l_gflops_eval = 1.0E-9 * l_num_flops / l_time_eval;
  l_time_total = l_time_compile_ts + l_time_eval;
  l_gflops_total = 1.0E-9 * l_num_flops / (l_time_total);

  std::cout << "  #flops:         " << l_num_flops       << std::endl;
  std::cout << "  time (compile): " << l_time_compile_ts << std::endl;
  std::cout << "  time (eval):    " << l_time_eval       << std::endl;
  std::cout << "  gflops (eval):  " << l_gflops_eval     << std::endl;
  std::cout << "  gflops (total): " << l_gflops_total    << std::endl;

  std::cout << "finished running bench_resnet" << std::endl;

  return EXIT_SUCCESS;
}