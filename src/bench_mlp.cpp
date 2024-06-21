#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include "src/backend/EinsumNode.h"

int main( int     i_argc,
          char  * i_argv[] ) {
  if( i_argc < 2 ) {
    std::cerr << "usage: ./bench_mlp path_to_model.pt store_and_lock" << std::endl;
    std::cerr << "store_and_lock is optional (defaults to 0) "
              << "and will exclude the initial transpose from the time measurements if 1"
              << std::endl;
    return EXIT_FAILURE;
  }
  std::string l_model_path( i_argv[1] );

  bool l_store_and_lock = false;
  if( i_argc > 2 ) {
    int l_arg_sl = std::stoi( i_argv[2] );

    if( l_arg_sl == 1 ) {
      l_store_and_lock = true;
    }
  }

  std::cout << "running MLP using" << std::endl;
  std::cout << "  model path: " << l_model_path << std::endl;
  std::cout << "  store and lock: " << l_store_and_lock << std::endl;

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
    std::cout << "  name:        "  << l_par.name << std::endl;
    std::cout << "  n_dimension: " << l_par.value.ndimension() << std::endl;
    std::cout << "  sizes:       " << l_par.value.sizes() << std::endl;
    std::cout << "  type:        "  << l_par.value.dtype() << std::endl;
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
  at::Tensor l_data = at::randn( { 1152, 784 } );

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
  at::Tensor l_out = at::rand( { 1152, 10 } );

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  0, 18 ) ); // batch 0
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  1, 64 ) ); // batch 1

  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  2,  8 ) ); // features 00
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  3, 98 ) ); // features 01

  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  4,  8 ) ); // features 10
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  5, 64 ) ); // features 11

  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  6,  8 ) ); // features 20
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  7, 64 ) ); // features 21

  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  8,  8 ) ); // features 30
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  9, 64 ) ); // features 31

  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 10,  8 ) ); // features 40
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 11, 64 ) ); // features 41

  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 12, 10 ) ); // features 50

  std::map< int64_t, int64_t > l_dim_sizes_aux;
  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >(  0,  1 ) ); // batch 0
  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >(  1,  1 ) ); // batch 1

  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >(  4,  8 ) ); // features 10
  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >(  5, 64 ) ); // features 11

  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >(  6,  8 ) ); // features 20
  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >(  7, 64 ) ); // features 21

  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >(  8,  8 ) ); // features 30
  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >(  9, 64 ) ); // features 31

  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >( 10,  8 ) ); // features 40
  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >( 11, 64 ) ); // features 41

  l_dim_sizes_aux.insert( std::pair< int64_t, int64_t >( 12, 10 ) ); // features 50

  int64_t l_dim_ids_input[4]    = {  0,  1,  2,  3 };
  int64_t l_dim_ids_hidden_0[4] = {  0,  1,  4,  5 };
  int64_t l_dim_ids_hidden_1[4] = {  0,  1,  6,  7 };
  int64_t l_dim_ids_hidden_2[4] = {  0,  1,  8,  9 };
  int64_t l_dim_ids_hidden_3[4] = {  0,  1, 10, 11 };
  int64_t l_dim_ids_out[3]      = {  0,  1, 12 };

  int64_t l_dim_ids_weight_0[4] = {  4,  5,  2,  3 };
  int64_t l_dim_ids_weight_1[4] = {  6,  7,  4,  5 };
  int64_t l_dim_ids_weight_2[4] = {  8,  9,  6,  7 };
  int64_t l_dim_ids_weight_3[4] = { 10, 11,  8,  9 };
  int64_t l_dim_ids_weight_4[4] = {     12, 10, 11 };

  einsum_ir::backend::EinsumNode l_node_input;
  einsum_ir::backend::EinsumNode l_node_hidden_0;
  einsum_ir::backend::EinsumNode l_node_hidden_1;
  einsum_ir::backend::EinsumNode l_node_hidden_2;
  einsum_ir::backend::EinsumNode l_node_hidden_3;
  einsum_ir::backend::EinsumNode l_node_out;

  einsum_ir::backend::EinsumNode l_node_weight_0;
  einsum_ir::backend::EinsumNode l_node_weight_1;
  einsum_ir::backend::EinsumNode l_node_weight_2;
  einsum_ir::backend::EinsumNode l_node_weight_3;
  einsum_ir::backend::EinsumNode l_node_weight_4;

  einsum_ir::backend::MemoryManager l_memory;

  l_node_input.init( 4,
                     l_dim_ids_input,
                     &l_dim_sizes,
                     nullptr,
                     einsum_ir::FP32,
                     l_data.data_ptr(),
                     &l_memory );

  l_node_weight_0.init( 4,
                        l_dim_ids_weight_0,
                        &l_dim_sizes,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_weights[0].data_ptr(),
                        &l_memory );

  l_node_weight_1.init( 4,
                        l_dim_ids_weight_1,
                        &l_dim_sizes,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_weights[1].data_ptr(),
                        &l_memory );

  l_node_weight_2.init( 4,
                        l_dim_ids_weight_2,
                        &l_dim_sizes,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_weights[2].data_ptr(),
                        &l_memory );

  l_node_weight_3.init( 4,
                        l_dim_ids_weight_3,
                        &l_dim_sizes,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_weights[3].data_ptr(),
                        &l_memory );

  l_node_weight_4.init( 3,
                        l_dim_ids_weight_4,
                        &l_dim_sizes,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_weights[4].data_ptr(),
                        &l_memory );

  l_node_hidden_0.init( 4,
                        l_dim_ids_hidden_0,
                        &l_dim_sizes,
                        &l_dim_sizes_aux,
                        nullptr,
                        nullptr,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_biases[0].data_ptr(),
                        nullptr,
                        einsum_ir::kernel_t::COPY,
                        einsum_ir::kernel_t::MADD,
                        einsum_ir::kernel_t::RELU,
                        &l_node_input,
                        &l_node_weight_0,
                        &l_memory );

  l_node_hidden_1.init( 4,
                        l_dim_ids_hidden_1,
                        &l_dim_sizes,
                        &l_dim_sizes_aux,
                        nullptr,
                        nullptr,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_biases[1].data_ptr(),
                        nullptr,
                        einsum_ir::kernel_t::COPY,
                        einsum_ir::kernel_t::MADD,
                        einsum_ir::kernel_t::RELU,
                        &l_node_hidden_0,
                        &l_node_weight_1,
                        &l_memory );

  l_node_hidden_2.init( 4,
                        l_dim_ids_hidden_2,
                        &l_dim_sizes,
                        &l_dim_sizes_aux,
                        nullptr,
                        nullptr,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_biases[2].data_ptr(),
                        nullptr,
                        einsum_ir::kernel_t::COPY,
                        einsum_ir::kernel_t::MADD,
                        einsum_ir::kernel_t::RELU,
                        &l_node_hidden_1,
                        &l_node_weight_2,
                        &l_memory );

  l_node_hidden_3.init( 4,
                        l_dim_ids_hidden_3,
                        &l_dim_sizes,
                        &l_dim_sizes_aux,
                        nullptr,
                        nullptr,
                        nullptr,
                        einsum_ir::FP32,
                        l_fc_biases[3].data_ptr(),
                        nullptr,
                        einsum_ir::kernel_t::COPY,
                        einsum_ir::kernel_t::MADD,
                        einsum_ir::kernel_t::RELU,
                        &l_node_hidden_2,
                        &l_node_weight_3,
                        &l_memory );

  l_node_out.init( 3,
                   l_dim_ids_out,
                   &l_dim_sizes,
                   &l_dim_sizes_aux,
                   nullptr,
                   nullptr,
                   nullptr,
                   einsum_ir::FP32,
                   l_fc_biases[4].data_ptr(),
                   l_out.data_ptr(),
                   einsum_ir::kernel_t::COPY,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &l_node_hidden_3,
                   &l_node_weight_4,
                   &l_memory );

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
  l_node_weight_3.store_and_lock_data();
  l_node_weight_4.store_and_lock_data();
  if( l_store_and_lock ) {
    l_node_input.store_and_lock_data();
  }

  // enable threading
#ifdef _OPENMP
  // four times overload
  int64_t l_num_tasks = omp_get_max_threads() * 4;

  l_node_hidden_0.threading_intra_op( l_num_tasks );
  l_node_hidden_1.threading_intra_op( l_num_tasks );
  l_node_hidden_2.threading_intra_op( l_num_tasks );
  l_node_hidden_3.threading_intra_op( l_num_tasks );
  l_node_out.threading_intra_op( l_num_tasks );
#endif

  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  // warm up
  l_node_out.eval();

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
  std::cout << "CSV_DATA: "
            << "einsum_ir,"
            << "\"" << l_model_path << "\","
            << l_num_flops << ","
            << l_time_compile << ","
            << l_time_eval << ","
            << l_gflops_eval << ","
            << l_gflops_total
            << std::endl;

  /*
   * torchscript model
   */
  std::cout << "running torchscript model" << std::endl;

  l_tp0 = std::chrono::steady_clock::now();
  l_model = torch::jit::optimize_for_inference( l_model );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  // warm up
  at::Tensor l_out_torch = l_model.forward( { l_data } ).toTensor();

  l_tp0 = std::chrono::steady_clock::now();
  l_model.forward( { l_data } );
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_eval = l_dur.count();

  l_gflops_eval = 1.0E-9 * l_num_flops / l_time_eval;
  l_time_total = l_time_compile + l_time_eval;
  l_gflops_total = 1.0E-9 * l_num_flops / (l_time_total);

  std::cout << "  #flops:         " << l_num_flops   << std::endl;
  std::cout << "  time (compile): " << l_time_compile << std::endl;
  std::cout << "  time (eval):    " << l_time_eval    << std::endl;
  std::cout << "  gflops (eval):  " << l_gflops_eval  << std::endl;
  std::cout << "  gflops (total): " << l_gflops_total << std::endl;
  std::cout << "CSV_DATA: "
            << "torch::jit::script,"
            << "\"" << l_model_path << "\","
            << l_num_flops << ","
            << l_time_compile << ","
            << l_time_eval << ","
            << l_gflops_eval << ","
            << l_gflops_total
            << std::endl;

  /*
   * compare solutions
   */
  if( !at::allclose( l_out_torch, l_out, 1E-3, 1E-4 ) ) {
    std::cerr << "error: einsum_ir solution is not close to torch!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}