#include <iostream>
#include <string>

#include <ATen/ATen.h>
#include "frontend/EinsumTree.h"
#include "frontend/EinsumTreeAscii.h"

int main( int     i_argc,
          char  * i_argv[] ) {
  if( i_argc < 3 ) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  ./bench_tree einsum_tree dimension_sizes dtype" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  * einsum_tree:      A compiled einsum tree." << std::endl;
    std::cerr << "  * dimension_sizes:  Dimension sizes have to be in ascending order of the dimension ids." << std::endl;
    std::cerr << "  * dtype:            FP32, FP64, default: FP32." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example:" << std::endl;
    std::cerr << "  ./bench_tree \"[[3,0]->[0,3]],[[3,2,4],[1,4,2]->[1,2,3]]->[0,1,2]\" \"2,3,4,5,6\" FP32" << std::endl;
    return EXIT_FAILURE;
  }

  std::string l_expression_string_arg( i_argv[1] );


  /*
   * count number of nodes
   */
  int64_t l_num_nodes = einsum_ir::frontend::EinsumTreeAscii::count_nodes( l_expression_string_arg );


  /*
   * initialize data structures
   */
  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  std::vector< std::vector< int64_t > > l_children;
  std::vector< std::vector< int64_t > > l_dim_ids;
  std::map< int64_t, int64_t > l_map_dim_sizes;
  l_dim_ids.resize(  l_num_nodes );
  l_children.resize( l_num_nodes );


  /*
   * parse expression string
   */
  int64_t l_analyzed_nodes = 0;
  l_err = einsum_ir::frontend::EinsumTreeAscii::parse_tree( l_expression_string_arg,
                                                            l_dim_ids, 
                                                            l_children,
                                                            l_analyzed_nodes );
  if( l_err != einsum_ir::SUCCESS ||
      l_num_nodes != l_analyzed_nodes ) {
    std::cerr << "error: failed to parse einsum tree" << std::endl;
    return EXIT_FAILURE;
  }


  /*
   * parse dimension sizes
   */
  std::string l_dim_sizes_arg( i_argv[2] );
  einsum_ir::frontend::EinsumTreeAscii::parse_dim_size( l_dim_sizes_arg,
                                                        l_dim_ids,
                                                        l_map_dim_sizes );
  
  /*
   * parse dtype
   */
  at::ScalarType l_dtype_at = at::ScalarType::Float;
  einsum_ir::data_t l_dtype_einsum_ir = einsum_ir::FP32;
  if( i_argc > 3 ) {
    std::string l_dtype_arg( i_argv[3] );
    if( l_dtype_arg == "FP64"){
      l_dtype_at = at::ScalarType::Double;
      l_dtype_einsum_ir = einsum_ir::FP64;
    }
  }

  /*
   * create external tensors
   */
  std::vector< at::Tensor > l_data;
  std::vector< void * > l_data_ptrs;
  for( int64_t l_id = 0; l_id < l_num_nodes; l_id++ ){
    //node needs an external tensor
    if( l_children[l_id].size() == 0 || l_id == l_num_nodes - 1 ){
      std::vector< int64_t > l_sizes;
      for( size_t l_di = 0; l_di < l_dim_ids[l_id].size(); l_di++ ) {
        int64_t l_dim_id = l_dim_ids[l_id][l_di];
        int64_t l_size   = l_map_dim_sizes[ l_dim_id ];
        l_sizes.push_back( l_size );
      }
      l_data.push_back( at::randn( l_sizes, l_dtype_at ) );
      l_data_ptrs.push_back( l_data.back().data_ptr() );
    }
    //node does not need an external tensor
    else{
      l_data_ptrs.push_back( nullptr );
    }
  }


  /*
   * performance data structures
   */
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_num_flops = 0;
  double l_time_compile = 0;
  double l_time_eval = 0;
  double l_time_total = 0;
  double l_gflops_eval = 0;
  double l_gflops_total = 0;


  /*
   * run einsum_ir
   */
  std::cout <<  "\n*** benchmarking einsum_ir ***" << std::endl;

  
  einsum_ir::frontend::EinsumTree l_einsum_tree;
  l_einsum_tree.init( &l_dim_ids,
                      &l_children,
                      &l_map_dim_sizes,
                      l_dtype_einsum_ir,
                      l_data_ptrs.data() );

  l_tp0 = std::chrono::steady_clock::now();
  l_err = l_einsum_tree.compile();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  if( l_err != einsum_ir::SUCCESS ) {
    std::cerr << "error: failed to compile einsum_ir tree" << std::endl;
    return EXIT_FAILURE;
  }

  // warmup run
  l_einsum_tree.eval();

  l_tp0 = std::chrono::steady_clock::now();
  l_einsum_tree.eval();
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_eval = l_dur.count();

  l_num_flops = l_einsum_tree.num_ops();
  l_gflops_eval = 1.0E-9 * l_num_flops / l_time_eval;
  l_time_total = l_time_compile + l_time_eval;
  l_gflops_total = 1.0E-9 * l_num_flops / (l_time_total);

  std::cout << "  #flops:         " << l_num_flops << std::endl;\
  std::cout << "  time (compile): " << l_time_compile << std::endl;
  std::cout << "  time (eval):    " << l_time_eval << std::endl;
  std::cout << "  gflops (eval):  " << l_gflops_eval << std::endl;
  std::cout << "  gflops (total): " << l_gflops_total << std::endl;
  std::cout << "CSV_DATA: "
            << "einsum_ir,"
            << "\"" << l_expression_string_arg << "\","
            << "\"" << l_dim_sizes_arg << "\","
            << l_num_flops << ","
            << l_time_compile << ","
            << l_time_eval << ","
            << l_gflops_eval << ","
            << l_gflops_total
            << std::endl;
  
}

