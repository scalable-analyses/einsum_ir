#include <iostream>
#include <string>

#include <ATen/ATen.h>
#include "frontend/EinsumExpression.h"
#include "frontend/EinsumExpressionAscii.h"

int main( int     i_argc,
          char  * i_argv[] ) {
  if( i_argc < 4 ) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  bench_expression einsum_string dimension_sizes contraction_path dtype store_lock print_tree" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  * dimension_sizes:  Dimension sizes have to be in ascending order of the dimension ids." << std::endl;
    std::cerr << "  * contraction_path: Contraction path." << std::endl;
    std::cerr << "  * dtype:            FP32, FP64, CPX_FP32 or CPX_FP64, default: FP32." << std::endl;
    std::cerr << "  * store_lock:       If 1 all einsum_ir input tensors are stored and locked before evaluation, default: 0." << std::endl;
    std::cerr << "  * print_tree:       If not 0 the einsum tree is printed (1: dimension ids, 2: characters), default: 0." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example:" << std::endl;
    std::cerr << "  ./bench_expression \"iae,bf,dcba,cg,dh->hgfei\" \"32,8,4,2,16,64,8,8,8\" \"(1,2),(2,3),(0,1),(0,1)\"" << std::endl;
    return EXIT_FAILURE;
  }

  /**
   * parse input tensors and output tensors
   **/
  std::string l_expression_string( i_argv[1] );
  std::vector< std::string > l_tensors;
  einsum_ir::frontend::EinsumExpressionAscii::parse_tensors( l_expression_string,
                                                             l_tensors );
  int64_t l_num_tensors = l_tensors.size();

  std::cout << "parsed tensors:" << std::endl;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::cout << "  " << l_tensors[l_te] << std::endl;
  }

  /*
   * parse dimension sizes
   */
  std::string l_dim_sizes_string( i_argv[2] );
  std::vector< int64_t > l_dim_sizes;
  einsum_ir::frontend::EinsumExpressionAscii::parse_dim_sizes( l_dim_sizes_string,
                                                               l_dim_sizes );

  /*
   * parse contraction path
   */
  std::string l_path_string( i_argv[3] );
  std::vector< int64_t > l_path;
  einsum_ir::frontend::EinsumExpressionAscii::parse_path( l_path_string,
                                                          l_path );

  std::cout << "parsed contraction path: ";
  for( std::size_t l_co = 0; l_co < l_path.size(); l_co++ ) {
    std::cout << l_path[l_co] << " ";
  }
  std::cout << std::endl;

  /*
   * create mapping from dimension name to id
   */
  std::map< char, int64_t > m_map_dim_name_to_id;
  einsum_ir::frontend::EinsumExpressionAscii::parse_dim_ids( l_expression_string,
                                                             m_map_dim_name_to_id );

  std::cout << "parsed dimension ids:" << std::endl;
  for( std::map< char, int64_t >::iterator l_di = m_map_dim_name_to_id.begin(); l_di != m_map_dim_name_to_id.end(); l_di++ ) {
    char l_dim_name = l_di->first;
    int64_t l_dim_id = l_di->second;

    std::cout << "  " << l_dim_name << ": " <<  l_dim_id << std::endl;
  }

  std::cout << "parsed dimension sizes:" << std::endl;
  // iterate over keys of map dim name to id
  for( std::map< char, int64_t >::iterator l_di = m_map_dim_name_to_id.begin(); l_di != m_map_dim_name_to_id.end(); l_di++ ) {
    char l_dim_name = l_di->first;
    int64_t l_dim_id = l_di->second;
    int64_t l_dim_size = l_dim_sizes[ l_dim_id ];

    std::cout << "  " << l_dim_name << ": " <<  l_dim_size << std::endl;
  }

  int64_t l_cpx_batch_dim_id = m_map_dim_name_to_id.size();

  /*
   * parse ctype and dtype
   */
  at::ScalarType l_dtype_at = at::ScalarType::Float;
  einsum_ir::complex_t l_ctype_einsum_ir = einsum_ir::REAL_ONLY;
  einsum_ir::data_t    l_dtype_einsum_ir = einsum_ir::FP32;
  if( i_argc > 4 ) {
    std::string l_arg_dtype = std::string( i_argv[4] );

    einsum_ir::frontend::EinsumExpressionAscii::parse_dtype( l_arg_dtype,
                                                             l_dtype_einsum_ir );
    einsum_ir::frontend::EinsumExpressionAscii::parse_ctype( l_arg_dtype,
                                                             l_ctype_einsum_ir );

    if( l_arg_dtype == "FP32" ) {
      l_dtype_at = at::ScalarType::Float;
    }
    else if( l_arg_dtype == "FP64" ) {
      l_dtype_at = at::ScalarType::Double;
    }
    else if( l_arg_dtype == "CPX_FP32" ) {
      l_dtype_at = at::ScalarType::ComplexFloat;
    }
    else if( l_arg_dtype == "CPX_FP64" ) {
      l_dtype_at = at::ScalarType::ComplexDouble;
    }
  }

  if( l_ctype_einsum_ir == einsum_ir::REAL_ONLY ) {
    std::cout << "ctype: REAL_ONLY" << std::endl;
  }
  else if( l_ctype_einsum_ir == einsum_ir::BATCH_INNER ) {
    std::cout << "ctype: BATCH_INNER" << std::endl;
  }
  else {
    std::cerr << "failed to determine ctype" << std::endl;
    return EXIT_FAILURE;
  }

  if( l_dtype_einsum_ir == einsum_ir::FP32 ) {
    std::cout << "dtype: FP32" << std::endl;
  }
  else if( l_dtype_einsum_ir == einsum_ir::FP64 ) {
    std::cout << "dtype: FP64" << std::endl;
  }
  else {
    std::cerr << "failed to determine dtype" << std::endl;
    return EXIT_FAILURE;
  }

  if( l_ctype_einsum_ir != einsum_ir::REAL_ONLY ) {
    l_dim_sizes.push_back( 2 );
  }

  /*
   * parse store_lock
   */
  bool l_store_and_lock = false;
  if( i_argc > 5 ) {
    int l_arg_sl = std::stoi( i_argv[5] );
    if( l_arg_sl == 1 ) {
      l_store_and_lock = true;
    }
  }
  std::cout << "store and lock: " << l_store_and_lock << std::endl;

  /*
   * parse print_tree
   */
  int64_t l_print_tree = 0;
  if( i_argc > 6 ) {
    l_print_tree = std::stoi( i_argv[6] );
  }
  if( l_print_tree < 0 || l_print_tree > 2 ) {
    std::cerr << "error: invalid print_tree argument" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "print_tree: " << l_print_tree << std::endl;

  /*
   * assemble einsum_ir data structures
   */
  std::vector< int64_t > l_string_num_dims( l_num_tensors );
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    l_string_num_dims[l_te] = l_tensors[l_te].size();
    if( l_ctype_einsum_ir != einsum_ir::REAL_ONLY ) {
      l_string_num_dims[l_te] += 1;
    }
  }

  std::vector< int64_t > l_string_dim_ids;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::string l_tensor = l_tensors[l_te];

    for( std::size_t l_ch = 0; l_ch < l_tensor.size(); l_ch++ ) {
      int64_t l_dim_id = m_map_dim_name_to_id[ l_tensor[l_ch] ];
      l_string_dim_ids.push_back( l_dim_id );
    }
    if( l_ctype_einsum_ir != einsum_ir::REAL_ONLY ) {
      l_string_dim_ids.push_back( l_cpx_batch_dim_id );
    }
  }

  std::cout << "assembled einsum_ir data structures" << std::endl;
  std::cout << "  string_num_dims: ";
  for( std::size_t l_te = 0; l_te < l_string_num_dims.size(); l_te++ ) {
    std::cout << l_string_num_dims[l_te] << " ";
  }
  std::cout << std::endl;

  std::cout << "  string_dim_ids: ";
  for( std::size_t l_ch = 0; l_ch < l_string_dim_ids.size(); l_ch++ ) {
    std::cout << l_string_dim_ids[l_ch] << " ";
  }
  std::cout << std::endl;

  /*
   * create the tensors' data
   */
  std::vector< at::Tensor > l_data;
  int64_t l_off = 0;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    // assemble size of the tensor
    std::vector< int64_t > l_sizes;
    int64_t l_num_dims = l_string_num_dims[l_te];
    if( l_ctype_einsum_ir != einsum_ir::REAL_ONLY ) {
      l_num_dims -= 1;
    }
    for( int64_t l_di = 0; l_di < l_num_dims; l_di++ ) {
      int64_t l_dim_id = l_string_dim_ids[l_off + l_di];
      int64_t l_size = l_dim_sizes[ l_dim_id ];
      l_sizes.push_back( l_size );
    }
    l_off += l_string_num_dims[l_te];

    l_data.push_back( at::randn( l_sizes, l_dtype_at ) );
  }

  std::vector< void * > l_data_ptrs;
  for( std::size_t l_te = 0; l_te < l_data.size(); l_te++ ) {
    l_data_ptrs.push_back( l_data[l_te].data_ptr() );
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

  einsum_ir::frontend::EinsumExpression l_einsum_exp;
  l_einsum_exp.init( l_dim_sizes.size(),
                     l_dim_sizes.data(),
                     l_path.size()/2,
                     l_string_num_dims.data(),
                     l_string_dim_ids.data(),
                     l_path.data(),
                     l_ctype_einsum_ir,
                     l_dtype_einsum_ir,
                     l_data_ptrs.data() );

  l_tp0 = std::chrono::steady_clock::now();
  einsum_ir::err_t l_err = l_einsum_exp.compile();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  if( l_err != einsum_ir::SUCCESS ) {
    std::cerr << "error: failed to compile einsum_ir expression" << std::endl;
    return EXIT_FAILURE;
  }

  // print einsum tree
  std::string l_tree = "";
  if( l_print_tree != 0 ) {
    l_tree = l_einsum_exp.to_string();
  }
  if( l_print_tree == 1 ) {
    std::cout << std::endl << l_tree;
  }
  else if( l_print_tree == 2 ) {
    // replace dimension ids with names (descending order)
    for( std::map< char, int64_t >::reverse_iterator l_di = m_map_dim_name_to_id.rbegin(); l_di != m_map_dim_name_to_id.rend(); l_di++ ) {
      char l_dim_name = l_di->first;
      int64_t l_dim_id = l_di->second;

      std::string l_dim_id_str = std::to_string( l_dim_id );
      std::string l_dim_name_str( 1, l_dim_name );
      size_t l_pos = 0;
      while( (l_pos = l_tree.find( l_dim_id_str, l_pos )) != std::string::npos ) {
        l_tree.replace( l_pos, l_dim_id_str.size(), l_dim_name_str );
        l_pos += l_dim_name_str.size();
      }
    }
    std::cout << std::endl << l_tree;
  }

  // stage input tensors if requested
  if( l_store_and_lock ) {
    for( int64_t l_te = 0; l_te < l_num_tensors-1; l_te++ ) {
      l_err = l_einsum_exp.store_and_lock_data( l_te );
      if( l_err != einsum_ir::SUCCESS ) {
        std::cerr << "error: failed to store and lock tensor with id: " << l_te << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  // warmup run
  l_einsum_exp.eval();

  l_tp0 = std::chrono::steady_clock::now();
  l_einsum_exp.eval();
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_eval = l_dur.count();

  l_num_flops = l_einsum_exp.num_ops();
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
            << "\"" << l_expression_string << "\","
            << "\"" << l_dim_sizes_string << "\","
            << "\"" << l_path_string << "\","
            << l_num_flops << ","
            << l_time_compile << ","
            << l_time_eval << ","
            << l_gflops_eval << ","
            << l_gflops_total
            << std::endl;

  /*
   * run at::einsum
   */
  std::cout << "\n*** benchmarking at::einsum ***" << std::endl;
  l_time_compile = 0;
  l_time_eval = 0;
  l_time_total = 0;
  l_gflops_eval = 0;
  l_gflops_total = 0;

  std::vector< at::Tensor > l_data_in( l_num_tensors-1 );
  for( int64_t l_te = 0; l_te < l_num_tensors - 1; l_te++ ) {
    l_data_in[l_te] = l_data[l_te];
  }

  // warmup run
  at::Tensor l_out_aten = at::einsum( l_expression_string,
                                      l_data_in,
                                      l_path );

  l_tp0 = std::chrono::steady_clock::now();
  l_out_aten = at::einsum( l_expression_string,
                           l_data_in,
                           l_path );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_total = l_dur.count();
  l_gflops_total = 1.0E-9 * l_num_flops / (l_time_total);

  std::cout << "  #flops:         " << l_num_flops << std::endl;
  std::cout << "  time (total):   " << l_time_total << std::endl;
  std::cout << "  gflops (total): " << l_gflops_total << std::endl;
  std::cout << "CSV_DATA: "
            << "at::einsum,"
            << "\"" << l_expression_string << "\","
            << "\"" << l_dim_sizes_string << "\","
            << "\"" << l_path_string << "\","
            << l_num_flops << ","
            << l_time_compile << ","
            << l_time_eval << ","
            << l_gflops_eval << ","
            << l_gflops_total
            << std::endl;

  /*
   * run at::matmul
   */
  std::cout << "\n*** benchmarking at::matmul ***" << std::endl;
  l_time_compile = 0;
  l_time_eval = 0;
  l_time_total = 0;
  l_gflops_eval = 0;
  l_gflops_total = 0;
  int64_t l_num_flops_matmul = 0;

  // extract binary contractions
  einsum_ir::backend::EinsumNode * l_bin_conts = l_einsum_exp.m_nodes.data() + l_num_tensors-1;


  std::cout << "  C M N K for the binary contractions:" << std::endl;
  // iterate over binary contractions
  for( int64_t l_co = 0; l_co < l_num_tensors-2; l_co++ ) {
    // extract matrix sizes
    int64_t l_c = 1;
    int64_t l_m = 1;
    int64_t l_n = 1;
    int64_t l_k = 1;

    for( int64_t l_di = 0; l_di < l_bin_conts[l_co].m_cont->m_num_dims_c; l_di++ ) {
      l_c *= l_bin_conts[l_co].m_cont->m_sizes_c[l_di];
    }
    for( int64_t l_di = 0; l_di < l_bin_conts[l_co].m_cont->m_num_dims_m; l_di++ ) {
      l_m *= l_bin_conts[l_co].m_cont->m_sizes_m[l_di];
    }
    for( int64_t l_di = 0; l_di < l_bin_conts[l_co].m_cont->m_num_dims_n; l_di++ ) {
      l_n *= l_bin_conts[l_co].m_cont->m_sizes_n[l_di];
    }
    for( int64_t l_di = 0; l_di < l_bin_conts[l_co].m_cont->m_num_dims_k; l_di++ ) {
      l_k *= l_bin_conts[l_co].m_cont->m_sizes_k[l_di];
    }

    if( l_ctype_einsum_ir != einsum_ir::REAL_ONLY ) {
      l_c /= 2;
    }

    at::Tensor l_mat_a = at::randn( {l_c, l_k, l_m},
                                    l_dtype_at );
    at::Tensor l_mat_b = at::randn( {l_c, l_n, l_k},
                                    l_dtype_at );

    // warmup run
    at::Tensor l_mat_c = at::matmul( l_mat_b,
                                     l_mat_a );

    l_tp0 = std::chrono::steady_clock::now();
    l_mat_c = at::matmul( l_mat_b,
                          l_mat_a );
    l_tp1 = std::chrono::steady_clock::now();
    l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

    l_time_eval += l_dur.count();
    if( l_ctype_einsum_ir == einsum_ir::REAL_ONLY ) {
      l_num_flops_matmul += 2 * l_c * l_m * l_n * l_k - l_c * l_m * l_n;
    }
    else {
      l_num_flops_matmul += 4 * 2 * l_c * l_m * l_n * l_k - 2 * l_c * l_m * l_n;
    }

    std::cout << "    #" << l_co << ": "
              <<  l_c << " " << l_m << " " << l_n << " " << l_k
              << std::endl;
  }
  if( l_num_flops_matmul != l_num_flops ) {
    std::cerr << "error: flops performed through batched gemms dont match" << std::endl;
    return EXIT_FAILURE;
  }

  l_gflops_eval = 1.0E-9 * l_num_flops / (l_time_eval);

  std::cout << "  #flops:         " << l_num_flops << std::endl;
  std::cout << "  time (eval):    " << l_time_eval << std::endl;
  std::cout << "  gflops (eval):  " << l_gflops_eval << std::endl;
  std::cout << "CSV_DATA: "
            << "at::matmul,"
            << "\"" << l_expression_string << "\","
            << "\"" << l_dim_sizes_string << "\","
            << "\"" << l_path_string << "\","
            << l_num_flops << ","
            << l_time_compile << ","
            << l_time_eval << ","
            << l_gflops_eval << ","
            << l_gflops_total
            << std::endl;

  /*
   * compare solution
   */
  std::cout << std::endl;
  std::cout << "*** comparing solution ***:" << std::endl;
  std::cout << "  maximum absolute entry in ATen solution:      " << at::max( at::abs( l_out_aten ) ).item() << std::endl;
  std::cout << "  maximum absolute entry in einsum_ir solution: " << at::max( at::abs( l_data.back() ) ).item() << std::endl;
  std::cout << "  maximum element-wise difference:              " << at::max( at::abs( l_out_aten - l_data.back() ) ).item() << std::endl;
  if( !at::allclose( l_out_aten, l_data.back() ) ) {
    std::cerr << "warning: einsum_ir solution is not close to at:einsum!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}