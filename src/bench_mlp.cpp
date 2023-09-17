#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include "src/backend/EinsumNode.h"

int main() {
  std::string l_model_path = "model_mlp.pt";
  int l_num_batch = 64;

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

  std::vector< at::Tensor > l_fc_weights;
  std::vector< at::Tensor > l_fc_biases;

  auto l_named_params= l_model.named_parameters( true );
  for( const auto & l_par: l_named_params ) {
    std::cout << "  name: "  << l_par.name << std::endl;
    std::cout << "  n_dimension: " << l_par.value.ndimension() << std::endl;
    std::cout << "  sizes: " << l_par.value.sizes() << std::endl;
    std::cout << "  type: "  << l_par.value.dtype() << std::endl;
    std::cout << std::endl;

    if( l_par.name.find( "weight" ) != std::string::npos ) {
      l_fc_weights.push_back( l_par.value );
    }
    if( l_par.name.find( "bias" ) != std::string::npos ) {
      l_fc_biases.push_back( l_par.value );
    }
  }

  /*
   * performance data structures
   */
  at::Tensor l_data = at::rand( { l_num_batch, 784 } );

  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_num_flops = 0;
  double l_time_compile = 0;
  double l_time_eval = 0;
  double l_time_total = 0;
  double l_gflops_eval = 0;
  double l_gflops_total = 0;

  /*
   * einsum ir default
   */
  std::cout << "running einsum_ir model" << std::endl;
  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;

  // init einsum ir
  at::Tensor l_out = at::rand( { l_num_batch, 10 } );

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, l_num_batch ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 784 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 512 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 512 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4,  10 ) );

  int64_t l_dim_ids_input[2]    = { 0, 1 };
  int64_t l_dim_ids_hidden_0[2] = { 0, 2 };
  int64_t l_dim_ids_hidden_1[2] = { 0, 3 };
  int64_t l_dim_ids_out[2]      = { 0, 4 };

  int64_t l_dim_ids_weight_0[2] = { 2, 1 };
  int64_t l_dim_ids_weight_1[2] = { 3, 2 };
  int64_t l_dim_ids_weight_2[2] = { 4, 3 };

  einsum_ir::backend::EinsumNode l_node_input;
  einsum_ir::backend::EinsumNode l_node_hidden_0;
  einsum_ir::backend::EinsumNode l_node_hidden_1;
  einsum_ir::backend::EinsumNode l_node_out;

  einsum_ir::backend::EinsumNode l_node_weight_0;
  einsum_ir::backend::EinsumNode l_node_weight_1;
  einsum_ir::backend::EinsumNode l_node_weight_2;

  l_node_input.init( 2,
                     l_dim_ids_input,
                     &l_dim_sizes,
                     nullptr,
                     einsum_ir::FP32,
                     l_data.data_ptr() );

  l_node_weight_0.init( 2,
                        l_dim_ids_weight_0,
                        &l_dim_sizes,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_weights[0].data_ptr() );

  l_node_weight_1.init( 2,
                        l_dim_ids_weight_1,
                        &l_dim_sizes,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_weights[1].data_ptr() );

  l_node_weight_2.init( 2,
                        l_dim_ids_weight_2,
                        &l_dim_sizes,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_weights[2].data_ptr() );

  l_node_hidden_0.init( 2,
                        l_dim_ids_hidden_0,
                        &l_dim_sizes,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        einsum_ir::FP32,
                        nullptr,
                        nullptr,
                        einsum_ir::kernel_t::ZERO,
                        einsum_ir::kernel_t::MADD,
                        einsum_ir::kernel_t::RELU,
                        &l_node_input,
                        &l_node_weight_0 );

  l_node_hidden_1.init( 2,
                        l_dim_ids_hidden_1,
                        &l_dim_sizes,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        einsum_ir::FP32,
                        nullptr,
                        nullptr,
                        einsum_ir::kernel_t::ZERO,
                        einsum_ir::kernel_t::MADD,
                        einsum_ir::kernel_t::RELU,
                        &l_node_hidden_0,
                        &l_node_weight_1 );

  l_node_out.init( 2,
                   l_dim_ids_out,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   einsum_ir::FP32,
                   nullptr,
                   l_out.data_ptr(),
                   einsum_ir::kernel_t::ZERO,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &l_node_hidden_1,
                   &l_node_weight_2 );

  // compile and stage weights
  l_tp0 = std::chrono::steady_clock::now();

  l_err = l_node_out.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    std::cerr << "error: failed to compile MLP" << std::endl;
    return EXIT_FAILURE;
  }
  l_node_weight_0.store_and_lock_data();
  l_node_weight_1.store_and_lock_data();
  l_node_weight_2.store_and_lock_data();

  // enable threading
  l_node_hidden_0.threading_intra_op( 256 );
  l_node_hidden_1.threading_intra_op( 256 );
  l_node_out.threading_intra_op( 256 );
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  // run network
  l_tp0 = std::chrono::steady_clock::now();
  l_node_out.eval();
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_eval = l_dur.count();

  l_num_flops = l_node_out.num_ops();
  l_gflops_eval = 1.0E-9 * l_num_flops / l_time_eval;
  l_time_total = l_time_compile + l_time_eval;
  l_gflops_total = 1.0E-9 * l_num_flops / l_time_total;

  std::cout << "  #flops:         " << l_num_flops    << std::endl;
  std::cout << "  time (compile): " << l_time_compile << std::endl;
  std::cout << "  time (eval):    " << l_time_eval    << std::endl;
  std::cout << "  gflops (eval):  " << l_gflops_eval  << std::endl;
  std::cout << "  gflops (total): " << l_gflops_total << std::endl;

  /*
   * einsum_ir blocked
   */
  std::cout << "running blocked einsum_ir model" << std::endl;
  at::Tensor l_out_blocked = at::rand( { l_num_batch, 10 } );

  std::map< int64_t, int64_t > l_dim_sizes_blocked;
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  0, l_num_batch ) ); // 0
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  1,           2 ) ); // 1-0
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  2,           2 ) ); // 1-1
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  3,           2 ) ); // 1-2
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  4,          98 ) ); // 1-3
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  5,           2 ) ); // 2-0
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  6,           2 ) ); // 2-1
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  7,           2 ) ); // 2-2
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  8,          64 ) ); // 2-3
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >(  9,           2 ) ); // 3-0
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >( 10,           2 ) ); // 3-1
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >( 11,           2 ) ); // 3-2
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >( 12,          64 ) ); // 3-3
  l_dim_sizes_blocked.insert( std::pair< int64_t, int64_t >( 13,          10 ) ); // 4

  int64_t l_dim_ids_input_blocked[5]    = {  0,  1,  2,  3,  4 };
  int64_t l_dim_ids_hidden_0_blocked[5] = {  0,  5,  6,  7,  8 };
  int64_t l_dim_ids_hidden_1_blocked[5] = {  0,  9, 10, 11, 12 };
  int64_t l_dim_ids_out_blocked[2]      = {  0, 13 };

  int64_t l_dim_ids_weight_0_blocked[8] = {  5,  6,  7,  8,  1, 2, 3, 4 };
  int64_t l_dim_ids_weight_1_blocked[8] = {  9, 10, 11, 12,  5, 6, 7, 8 };
  int64_t l_dim_ids_weight_2_blocked[5] = { 13,  9, 10, 11, 12 };

  einsum_ir::backend::EinsumNode l_node_input_blocked;
  einsum_ir::backend::EinsumNode l_node_hidden_0_blocked;
  einsum_ir::backend::EinsumNode l_node_hidden_1_blocked;
  einsum_ir::backend::EinsumNode l_node_out_blocked;

  einsum_ir::backend::EinsumNode l_node_weight_0_blocked;
  einsum_ir::backend::EinsumNode l_node_weight_1_blocked;
  einsum_ir::backend::EinsumNode l_node_weight_2_blocked;

  l_node_input_blocked.init( 5,
                             l_dim_ids_input_blocked,
                             &l_dim_sizes_blocked,
                             nullptr,
                             einsum_ir::FP32,
                             l_data.data_ptr() );

  l_node_weight_0_blocked.init( 8,
                                l_dim_ids_weight_0_blocked,
                                &l_dim_sizes_blocked,
                                nullptr,
                                einsum_ir::FP32,
                                l_fc_weights[0].data_ptr() );

  l_node_weight_1_blocked.init( 8,
                                l_dim_ids_weight_1_blocked,
                                &l_dim_sizes_blocked,
                                nullptr,
                                einsum_ir::FP32,
                                l_fc_weights[1].data_ptr() );

  l_node_weight_2_blocked.init( 5,
                                l_dim_ids_weight_2_blocked,
                                &l_dim_sizes_blocked,
                                nullptr,
                                einsum_ir::FP32,
                                l_fc_weights[2].data_ptr() );

  l_node_hidden_0_blocked.init( 5,
                                l_dim_ids_hidden_0_blocked,
                                &l_dim_sizes_blocked,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                einsum_ir::FP32,
                                nullptr,
                                nullptr,
                                einsum_ir::kernel_t::ZERO,
                                einsum_ir::kernel_t::MADD,
                                einsum_ir::kernel_t::RELU,
                                &l_node_input_blocked,
                                &l_node_weight_0_blocked );

  l_node_hidden_1_blocked.init( 5,
                                l_dim_ids_hidden_1_blocked,
                                &l_dim_sizes_blocked,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                einsum_ir::FP32,
                                nullptr,
                                nullptr,
                                einsum_ir::kernel_t::ZERO,
                                einsum_ir::kernel_t::MADD,
                                einsum_ir::kernel_t::RELU,
                                &l_node_hidden_0_blocked,
                                &l_node_weight_1_blocked );

  l_node_out_blocked.init( 2,
                           l_dim_ids_out_blocked,
                           &l_dim_sizes_blocked,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           einsum_ir::FP32,
                           nullptr,
                           l_out_blocked.data_ptr(),
                           einsum_ir::kernel_t::ZERO,
                           einsum_ir::kernel_t::MADD,
                           einsum_ir::kernel_t::UNDEFINED_KTYPE,
                           &l_node_hidden_1_blocked,
                           &l_node_weight_2_blocked );

  // compile and stage weights
  l_tp0 = std::chrono::steady_clock::now();

  l_err = l_node_out_blocked.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    std::cerr << "error: failed to compile MLP" << std::endl;
    return EXIT_FAILURE;
  }
  l_node_weight_0_blocked.store_and_lock_data();
  l_node_weight_1_blocked.store_and_lock_data();
  l_node_weight_2_blocked.store_and_lock_data();

  // enable threading
  l_node_hidden_0_blocked.threading_intra_op( 256 );
  l_node_hidden_1_blocked.threading_intra_op( 256 );
  l_node_out_blocked.threading_intra_op( 256 );

  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  // run network
  l_tp0 = std::chrono::steady_clock::now();
  l_node_out_blocked.eval();
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_eval = l_dur.count();

  l_num_flops = l_node_out_blocked.num_ops();
  l_gflops_eval = 1.0E-9 * l_num_flops / l_time_eval;
  l_time_total = l_time_compile + l_time_eval;
  l_gflops_total = 1.0E-9 * l_num_flops / (l_time_total);

  std::cout << "  #flops:         " << l_num_flops    << std::endl;
  std::cout << "  time (compile): " << l_time_compile << std::endl;
  std::cout << "  time (eval):    " << l_time_eval    << std::endl;
  std::cout << "  gflops (eval):  " << l_gflops_eval  << std::endl;
  std::cout << "  gflops (total): " << l_gflops_total << std::endl;

  /*
   * torchscript model
   */
  std::cout << "running torchscript model" << std::endl;

  // dry run
  at::Tensor l_out_torch = l_model.forward( { l_data } ).toTensor();

  l_tp0 = std::chrono::steady_clock::now();
  l_model.forward( { l_data } );
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_eval = l_dur.count();

  l_gflops_eval = 1.0E-9 * l_num_flops / l_time_eval;
  l_time_total = l_time_compile + l_time_eval;
  l_gflops_total = 1.0E-9 * l_num_flops / (l_time_total);

  std::cout << "  #flops:         " << l_num_flops << std::endl;
  std::cout << "  time (total):   " << l_time_total << std::endl;
  std::cout << "  gflops (total): " << l_gflops_total << std::endl;

  /*
   * compare solutions
   */
  if( !at::allclose( l_out_torch, l_out, 1E-3, 1E-4 ) ) {
    std::cerr << "error: einsum_ir solution is not close to torch!" << std::endl;
    return EXIT_FAILURE;
  }

  if( !at::allclose( l_out_torch, l_out_blocked, 1E-3, 1E-4 ) ) {
    std::cerr << "error: blocked einsum_ir solution is not close to torch!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}