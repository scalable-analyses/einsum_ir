#include <iostream>
#include <string>

#include <ATen/ATen.h>
#include "frontend/EinsumExpression.h"

/**
 * Splits an input string by the given separation string.
 *
 * @param i_input input string.
 * @param i_separation separation string.
 * @param o_output output substrings.
 **/
void split_string( std::string                const & i_input,
                   std::string                const & i_separation,
                   std::vector< std::string >       & o_output ) {
  std::string l_string = i_input;
  int64_t l_off = 0;
  int64_t l_size_string = l_string.size();
  while( l_off < l_size_string ) {
    l_off = l_string.find( i_separation );
    if( l_off < 0 ) break;
    o_output.push_back( l_string.substr( 0, l_off ) );
    l_string.erase( 0, l_off + i_separation.size() );
  }
  if( l_string.size() > 0 ) {
    o_output.push_back( l_string );
  }
}

int main( int     i_argc,
          char  * i_argv[] ) {
  if( i_argc < 4 ) {
    std::cerr << "usage: bench_expression einsum_string dimension_sizes contraction_path dtype store_lock" << std::endl;
    std::cerr << "dimension sizes have to be in ascending order of the dimension ids" << std::endl;
    std::cerr << "dtype maybe be either FP32 or FP64, default: FP32" << std::endl;
    std::cerr << "if store_lock is 1, all einsum_ir input tensors are stored and locked before evaluation, default: 0" << std::endl;
    std::cerr << "example: ./bench_expression \"iae,bf,dcba,cg,dh->hgfei\" \"32,8,4,2,16,64,8,8,8\" \"(1,2),(2,3),(0,1),(0,1)\"" << std::endl;
    return EXIT_FAILURE;
  }

  /**
   * parse input tensors and output tensors
   **/
  std::string l_expression_string( i_argv[1] );
  l_expression_string.erase( std::remove( l_expression_string.begin(),
                                          l_expression_string.end(),
                                          ' '),
                             l_expression_string.end());
  std::vector< std::string > l_tensors_tmp;

  split_string( l_expression_string,
                std::string("->"),
                l_tensors_tmp );

  std::vector< std::string > l_tensors;
  split_string( l_tensors_tmp[0],
                std::string(","),
                l_tensors );
  l_tensors.push_back( l_tensors_tmp[1] );
  int64_t l_num_tensors = l_tensors.size();

  std::cout << "parsed tensors:" << std::endl;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::cout << "  " << l_tensors[l_te] << std::endl;
  }

  /*
   * parse dimension sizes
   */
  std::string l_dim_sizes_string( i_argv[2] );
  l_dim_sizes_string.erase( std::remove( l_dim_sizes_string.begin(),
                                         l_dim_sizes_string.end(),
                                         ' '),
                            l_dim_sizes_string.end());

  std::vector< std::string > l_dim_sizes_tmp;
  split_string( l_dim_sizes_string,
                std::string(","),
                l_dim_sizes_tmp );
  std::vector< int64_t > l_dim_sizes( l_dim_sizes_tmp.size() );
  for( std::size_t l_di = 0; l_di < l_dim_sizes_tmp.size(); l_di++ ) {
    l_dim_sizes[l_di] = std::stoi( l_dim_sizes_tmp[l_di] );
  }

  /*
   * parse contraction path
   */
  std::string l_path_string( i_argv[3] );
  l_path_string.erase( std::remove( l_path_string.begin(),
                                    l_path_string.end(),
                                    ' '),
                       l_path_string.end());
  l_path_string.erase( std::remove( l_path_string.begin(),
                                    l_path_string.end(),
                                    '('),
                       l_path_string.end());
  l_path_string.erase( std::remove( l_path_string.begin(),
                                    l_path_string.end(),
                                    ')'),
                       l_path_string.end());
  std::vector< std::string > l_path_tmp;
  split_string( l_path_string,
                std::string(","),
                l_path_tmp );
  std::vector< int64_t > l_path( l_path_tmp.size() );
  for( std::size_t l_co = 0; l_co < l_path_tmp.size(); l_co++ ) {
    l_path[l_co] = std::stoi( l_path_tmp[l_co] );
  }

  std::cout << "parsed contraction path: ";
  for( std::size_t l_co = 0; l_co < l_path.size(); l_co++ ) {
    std::cout << l_path[l_co] << " ";
  }
  std::cout << std::endl;

  /*
   * create mapping from dimension name to id
   */
  std::set< char > l_dim_names_set;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::string l_tensor = l_tensors[l_te];

    for( std::size_t l_ch = 0; l_ch < l_tensor.size(); l_ch++ ) {
      l_dim_names_set.insert( l_tensor[l_ch] );
    }
  }
  std::vector< char > l_dim_names( l_dim_names_set.begin(),
                                   l_dim_names_set.end() );

  std::cout << "parsed dimension sizes:" << std::endl;
  for( std::size_t l_di = 0; l_di < l_dim_sizes.size(); l_di++ ) {
    std::cout << "  " << l_dim_names[l_di] << ": " << l_dim_sizes[l_di] << std::endl;
  }

  std::map< char, int64_t > m_map_dim_name_to_id;
  for( std::size_t l_di = 0; l_di < l_dim_names.size(); l_di++ ) {
    m_map_dim_name_to_id.insert( { l_dim_names[l_di], l_di } );
  }

  /*
   * parse dtype
   */
  at::ScalarType l_dtype_at= at::ScalarType::Float;
  einsum_ir::data_t l_dtype_einsum_ir = einsum_ir::FP32;
  if( i_argc > 4 ) {
    std::string l_arg_dtype = std::string( i_argv[4] );
    if( l_arg_dtype == "FP64" ) {
      l_dtype_at = at::ScalarType::Double;
      l_dtype_einsum_ir = einsum_ir::FP64;
    }
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
   * assemble einsum_ir data structures
   */
  std::vector< int64_t > l_string_num_dims( l_num_tensors );
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    l_string_num_dims[l_te] = l_tensors[l_te].size();
  }

  std::vector< int64_t > l_string_dim_ids;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::string l_tensor = l_tensors[l_te];

    for( std::size_t l_ch = 0; l_ch < l_tensor.size(); l_ch++ ) {
      int64_t l_dim_id = m_map_dim_name_to_id[ l_tensor[l_ch] ];
      l_string_dim_ids.push_back( l_dim_id );
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
    for( int64_t l_di = 0; l_di < l_string_num_dims[l_te]; l_di++ ) {
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
                     l_dtype_einsum_ir,
                     l_data_ptrs.data() );

  l_tp0 = std::chrono::steady_clock::now();
  einsum_ir::err_t l_err = l_einsum_exp.compile();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  if( l_err != einsum_ir::SUCCESS ) {
    std::cerr << "error: failed to compile einsum_ir expressions" << std::endl;
    return EXIT_FAILURE;
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

  // dry run
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

  // dry run
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

    at::Tensor l_mat_a = at::rand( {l_c, l_k, l_m} );
    at::Tensor l_mat_b = at::rand( {l_c, l_n, l_k} );

    // dry run
    at::Tensor l_mat_c = at::matmul( l_mat_b,
                                     l_mat_a );

    l_tp0 = std::chrono::steady_clock::now();
    l_mat_c = at::matmul( l_mat_b,
                          l_mat_a );
    l_tp1 = std::chrono::steady_clock::now();
    l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

    l_time_eval += l_dur.count();
    l_num_flops_matmul += 2 * l_c * l_m * l_n * l_k - l_c * l_m * l_n;

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
  if( !at::allclose( l_out_aten, l_data.back() ) ) {
    std::cerr << "warning: einsum_ir solution is not close to at:einsum!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}