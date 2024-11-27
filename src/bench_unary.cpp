#include <cstdlib>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ATen/ATen.h>
#include "backend/UnaryTpp.h"

/**
 * Derives the permutation to transfer the input ids to the output ones.
 *
 * @param i_num_dims number of dimensions.
 * @param i_dim_ids_in dimension ids of the output tensor.
 * @param i_dim_ids_out dimension ids of the output tensor.
 * @param o_permutation will be set to derived permutation.
 **/
void permutation( int64_t         i_num_dims,
                  int64_t const * i_dim_ids_in,
                  int64_t const * i_dim_ids_out,
                  int64_t       * o_permutation ) {
  for( int64_t l_di_in = 0; l_di_in < i_num_dims; l_di_in++ ) {
    for( int64_t l_di_out = 0; l_di_out < i_num_dims; l_di_out++ ) {
      if( i_dim_ids_in[l_di_in] == i_dim_ids_out[l_di_out] ) {
        o_permutation[l_di_out] = l_di_in;
      }
    }
  }
}

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
  if( i_argc < 3 ) {
    std::cerr << "usage: bench_unary einsum_string dimension_sizes dtype"               << std::endl;
    std::cerr << "  dimension sizes have to be in ascending order of the dimension ids" << std::endl;
    std::cerr << "  dtype maybe be either FP32 or FP64, default: FP32"                  << std::endl;
    std::cerr << "example: ./bench_unary \"abcd->dbac\" \"32,32,32,32\" \"FP64\""       << std::endl;
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
  std::vector< std::string > l_tensors;

  split_string( l_expression_string,
                std::string("->"),
                l_tensors );

  std::cout << "parsed tensors:" << std::endl;
  std::cout << "  " << l_tensors[0] << std::endl;
  std::cout << "  " << l_tensors[1] << std::endl;

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
   * create mapping from dimension name to id
   */
  std::set< char > l_dim_names_set;
  for( int64_t l_te = 0; l_te < 2; l_te++ ) {
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
  if( i_argc > 3 ) {
    std::string l_arg_dtype = std::string( i_argv[3] );
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
    l_dtype_einsum_ir = einsum_ir::FP32;
  }

  if( l_tensors[0].size() != l_tensors[1].size() ) {
    std::cerr << "error: number of tensor dimensions does not match" << std::endl;
    return EXIT_FAILURE;
  }

  int64_t l_num_dims = l_tensors[0].size();

  /*
   * assemble einsum_ir data structures
   */
  std::vector< int64_t > l_string_dim_ids[2];
  for( int64_t l_te = 0; l_te < 2; l_te++ ) {
    std::string l_tensor = l_tensors[l_te];

    for( std::size_t l_ch = 0; l_ch < l_tensor.size(); l_ch++ ) {
      int64_t l_dim_id = m_map_dim_name_to_id[ l_tensor[l_ch] ];
      l_string_dim_ids[l_te].push_back( l_dim_id );
    }
  }

  std::cout << "assembled einsum_ir data structures" << std::endl;
  std::cout << "  string_dim_ids (input):  ";
  for( std::size_t l_ch = 0; l_ch < l_string_dim_ids[0].size(); l_ch++ ) {
    std::cout << l_string_dim_ids[0][l_ch] << " ";
  }
  std::cout << std::endl;

  std::cout << "  string_dim_ids (output): ";
  for( std::size_t l_ch = 0; l_ch < l_string_dim_ids[1].size(); l_ch++ ) {
    std::cout << l_string_dim_ids[1][l_ch] << " ";
  }
  std::cout << std::endl;

  /**
   * assemble aten data structure 
   **/
  std::vector< int64_t > l_permutation( l_num_dims );
  permutation( l_num_dims,
               l_string_dim_ids[0].data(),
               l_string_dim_ids[1].data(),
               l_permutation.data() );

  std::cout << "assembled aten data structure" << std::endl;
  std::cout << "  permutation: ";
  for( std::size_t l_di = 0; l_di < l_permutation.size(); l_di++ ) {
    std::cout << l_permutation[l_di] << " ";
  }
  std::cout << std::endl;

  /*
   * create the tensors' data
   */
  std::vector< at::Tensor > l_data;

  for( int64_t l_te = 0; l_te < 2; l_te++ ) {
    // assemble size of the tensor
    std::vector< int64_t > l_sizes;
    for( int64_t l_di = 0; l_di < l_num_dims; l_di++ ) {
      int64_t l_dim_id = l_string_dim_ids[l_te][l_di];
      int64_t l_size = l_dim_sizes[ l_dim_id ];
      l_sizes.push_back( l_size );
    }

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
  int64_t l_num_bytes;
  double l_time_compile = 0;
  double l_time_eval = 0;
  double l_time_total = 0;
  double l_gibs_eval = 0;
  double l_gibs_total = 0;

  l_num_bytes = 1;
  for( int64_t l_di = 0; l_di < l_num_dims; l_di++ ) {
    l_num_bytes *= l_dim_sizes[l_di];
  }
  if( l_dtype_einsum_ir == einsum_ir::data_t::FP32 ) {
    l_num_bytes *= 4;
  }
  else if( l_dtype_einsum_ir == einsum_ir::data_t::FP64 ) {
    l_num_bytes *= 8;
  }
  else {
    std::cerr << "error: unsupported dtype" << std::endl;
    return EXIT_FAILURE;
  }
  l_num_bytes *= 2; // read + write

  std::cout << "total amount of transferred data (read + write):"    << std::endl;
  std::cout << "  B:   " << l_num_bytes                              << std::endl;
  std::cout << "  MiB: " << l_num_bytes / (1024.0 * 1024.0)          << std::endl;
  std::cout << "  GiB: " << l_num_bytes / (1024.0 * 1024.0 * 1024.0) << std::endl;

  /*
   * einsum_ir
   */
  std::map< int64_t, int64_t > l_map_dim_sizes;
  for( int64_t l_di = 0; l_di < l_num_dims; l_di++ ) {
    l_map_dim_sizes.insert( std::pair< int64_t, int64_t >( l_di, l_dim_sizes[l_di] ) );
  }

  einsum_ir::backend::UnaryTpp l_unary_tpp;

  l_unary_tpp.init( l_num_dims,
                    &l_map_dim_sizes,
                    l_string_dim_ids[0].data(),
                    l_string_dim_ids[1].data(),
                    l_dtype_einsum_ir,
                    l_dtype_einsum_ir,
                    l_dtype_einsum_ir,
                    einsum_ir::kernel_t::COPY );

  l_tp0 = std::chrono::steady_clock::now();
  l_unary_tpp.compile();
  l_tp1 = std::chrono::steady_clock::now();

  // enable threading
#ifdef _OPENMP
  // four times overload
  int64_t l_num_tasks = omp_get_max_threads() * 4;

  l_unary_tpp.threading( l_num_tasks );
#endif

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  // warm up
  l_unary_tpp.eval( l_data_ptrs[0],
                    l_data_ptrs[1] );

  l_tp0 = std::chrono::steady_clock::now();
  l_unary_tpp.eval( l_data_ptrs[0],
                    l_data_ptrs[1] );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_eval = l_dur.count();

  l_time_total = l_time_compile + l_time_eval;

  l_gibs_eval = l_num_bytes;
  l_gibs_eval /= 1024.0 * 1024.0 * 1024.0;
  l_gibs_eval /= l_time_eval;

  l_gibs_total = l_num_bytes;
  l_gibs_total /= 1024.0 * 1024.0 * 1024.0;
  l_gibs_total /= l_time_total;

  std::cout << "einsum_ir:" << std::endl;
  std::cout << "  time (compile): " << l_time_compile << std::endl;
  std::cout << "  time (eval):    " << l_time_eval    << std::endl;
  std::cout << "  gibs (eval):    " << l_gibs_eval    << std::endl;
  std::cout << "  gibs (total):   " << l_gibs_total   << std::endl;

  /*
   * aten
   */
  l_tp0 = std::chrono::steady_clock::now();
  at::Tensor l_data_aten = l_data[0].permute( l_permutation ).contiguous();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_total = l_dur.count();

  l_gibs_total = l_num_bytes;
  l_gibs_total /= 1024.0 * 1024.0 * 1024.0;
  l_gibs_total /= l_time_total;

  std::cout << "at::permute:" << std::endl;
  std::cout << "  time (total):   " << l_time_total << std::endl;
  std::cout << "  gibs (total):   " << l_gibs_total << std::endl;

  /*
   * check correctness
   */
  if( !at::equal( l_data_aten, l_data[1] ) ) {
    std::cerr << "******************************************************" << std::endl;
    std::cerr << "* warning: aten solution is different from einsum_ir *" << std::endl;
    std::cerr << "******************************************************" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}