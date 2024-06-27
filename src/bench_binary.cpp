#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ATen/ATen.h>
#include "backend/BinaryContractionTpp.h"
#include "backend/EinsumNode.h"

void blocked_matmul() {
  std::cout << "*******************************" << std::endl;
  std::cout << "*** blocked matmul testcase ***" << std::endl;
  std::cout << "*******************************" << std::endl;

  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_n_flops = 0;
  int64_t l_repitions = 1;
  int64_t l_repitions_warm_up = 100;
  std::vector< int64_t > l_dim_ids_permute_left;
  std::vector< int64_t > l_dim_ids_permute_right;
  double l_time_compile = 0;
  double l_time = 0;
  double l_gflops = 0;
  
  //binary contraction parameter
  std::vector< int64_t > l_sizes_c = {512};
  std::vector< int64_t > l_sizes_m = {4, 16};
  std::vector< int64_t > l_sizes_n = {64};
  std::vector< int64_t > l_sizes_k = {2, 32};

  //                                          c0 m1 k1 m0 k0
  std::vector< int64_t> l_dim_ids_in_left  = { 0, 2, 5, 1, 4 };
  //                                          c0 k1 n0 k0
  std::vector< int64_t> l_dim_ids_in_right = { 0, 5, 3, 4 };
  //                                          c0 n0 m1 m0
  std::vector< int64_t> l_dim_ids_out      = { 0, 3, 2, 1 };

  l_dim_ids_permute_left  = {5, 4, 0, 2, 1};
  l_dim_ids_permute_right = {0, 3, 5, 4};

  //calculate sizes of equivalent matmul
  int64_t l_size_c = 1;
  int64_t l_size_m = 1;
  int64_t l_size_n = 1;
  int64_t l_size_k = 1;
  std::map< int64_t, int64_t > l_dim_sizes;
  int64_t l_dim_id = 0;
  for( int64_t l_id = 0; l_id < l_sizes_c.size(); l_id++ ){
    l_size_c *= l_sizes_c[l_id];
    l_dim_sizes.insert( std::pair< int64_t, int64_t >( l_dim_id, l_sizes_c[l_id] ) );
    l_dim_id++;
  }
  for( int64_t l_id = 0; l_id < l_sizes_m.size(); l_id++ ){
    l_size_m *= l_sizes_m[l_id];
    l_dim_sizes.insert( std::pair< int64_t, int64_t >( l_dim_id, l_sizes_m[l_id] ) );
    l_dim_id++;
  }
  for( int64_t l_id = 0; l_id < l_sizes_n.size(); l_id++ ){
    l_size_n *= l_sizes_n[l_id];
    l_dim_sizes.insert( std::pair< int64_t, int64_t >( l_dim_id, l_sizes_n[l_id] ) );
    l_dim_id++;
  }
  for( int64_t l_id = 0; l_id < l_sizes_k.size(); l_id++ ){
    l_size_k *= l_sizes_k[l_id];
    l_dim_sizes.insert( std::pair< int64_t, int64_t >( l_dim_id, l_sizes_k[l_id] ) );
    l_dim_id++;
  }

  // create vectors of sizes and einsum string
  std::vector< int64_t > l_sizes_left;
  std::vector< int64_t > l_sizes_right;
  std::vector< int64_t > l_sizes_out;
  std::string l_einsum_string;
  for( int64_t l_di = 0; l_di < l_dim_ids_in_left.size(); l_di++ ){
    l_sizes_left.push_back( l_dim_sizes[l_dim_ids_in_left[l_di]] );
    l_einsum_string +=  (char) 97 + l_dim_ids_in_left[l_di];
  }
  l_einsum_string += ","; 
  for( int64_t l_di = 0; l_di < l_dim_ids_in_right.size(); l_di++ ){
    l_sizes_right.push_back( l_dim_sizes[l_dim_ids_in_right[l_di]] );
    l_einsum_string +=  (char) 97 + l_dim_ids_in_right[l_di];
  }
  l_einsum_string += "->"; 
  for( int64_t l_di = 0; l_di < l_dim_ids_out.size(); l_di++ ){
    l_sizes_out.push_back( l_dim_sizes[l_dim_ids_out[l_di]] );
    l_einsum_string +=  (char) 97 + l_dim_ids_out[l_di];
  }

  //number of flops 
  l_n_flops = l_size_c * l_size_m * l_size_n * l_size_k * 2;

  /**
   * Matmul 
   **/
  at::Tensor l_mat_a = at::rand( { l_size_c, l_size_k, l_size_m } );
  at::Tensor l_mat_b = at::rand( { l_size_c, l_size_n, l_size_k } );
  at::Tensor l_out_matmul;
  std::cout << "matmul:" << std::endl;

  // warm up
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_rep = 0; l_rep < l_repitions_warm_up; l_rep++ ){
    l_out_matmul = at::matmul( l_mat_b, l_mat_a );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_repitions = l_repitions_warm_up / l_dur.count() + 1;

  // run with repititions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_rep = 0; l_rep < l_repitions; l_rep++ ){
    l_out_matmul = at::matmul( l_mat_b, l_mat_a );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count() / l_repitions;
  l_gflops = 1.0E-9 * l_n_flops / l_time;

  std::cout << "  time:   " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;

  /*
   * at::einsum
   */

  //create Tensors
  at::Tensor l_ten_left  = at::rand( at::IntArrayRef( l_sizes_left.data(),  l_sizes_left.size()  ) );
  at::Tensor l_ten_right = at::ones( at::IntArrayRef( l_sizes_right.data(), l_sizes_right.size() ) );
  at::Tensor l_ten_out   = at::rand( at::IntArrayRef( l_sizes_out.data(),   l_sizes_out.size()   ) );
  std::cout << "at::einsum:" << std::endl;

  // warm up
  at::Tensor l_ten_out_torch;
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_rep = 0; l_rep < l_repitions_warm_up; l_rep++ ){
    l_ten_out_torch = at::einsum( l_einsum_string,
                                  {l_ten_left, l_ten_right},
                                  { {0,1} } );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_repitions = l_repitions_warm_up / l_dur.count() + 1;

  // run with repititions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_rep = 0; l_rep < l_repitions; l_rep++ ){
    l_ten_out_torch = at::einsum( l_einsum_string,
                                  {l_ten_left, l_ten_right},
                                  { {0,1} } );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count() / l_repitions;
  l_gflops = 1.0E-9 * l_n_flops / l_time;

  std::cout << "  time (contract): " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;

  /*
   * einsum_ir
   */
  std::cout << "einsum_ir:" << std::endl;

  einsum_ir::backend::MemoryManager l_memory;

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( l_dim_ids_in_left.size(),
                   l_dim_ids_in_right.size(),
                   l_dim_ids_out.size(),
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_in_left.data(),
                   l_dim_ids_in_right.data(),
                   l_dim_ids_out.data(),
                   l_dim_ids_permute_left.data(),
                   l_dim_ids_permute_right.data(),
                   &l_memory,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  l_tp0 = std::chrono::steady_clock::now();
  l_bin_cont.compile();
  std::cout << "test" << std::endl;
  l_memory.alloc_all_memory();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  // enable threading
#ifdef _OPENMP
  // four times overload
  int64_t l_num_tasks = omp_get_max_threads() * 4;

  l_bin_cont.threading( l_num_tasks );
#endif

  // warm up
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_rep = 0; l_rep < l_repitions_warm_up; l_rep++ ){
    l_bin_cont.contract( l_ten_left.data_ptr(),
                        l_ten_right.data_ptr(),
                        l_ten_out.data_ptr() );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_repitions = l_repitions_warm_up / l_dur.count() + 1;

  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_rep = 0; l_rep < l_repitions; l_rep++ ){
    l_bin_cont.contract( l_ten_left.data_ptr(),
                        l_ten_right.data_ptr(),
                        l_ten_out.data_ptr() );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count() / l_repitions;
  l_gflops = 1.0E-9 * l_n_flops / l_time;

  std::cout << "  time (compile): " << l_time_compile << std::endl;
  std::cout << "  time (contract): " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;

  if( !at::allclose( l_ten_out_torch, l_ten_out, 1e-03 ) ) {
    std::cerr << "error: einsum_ir solution is not close to aten!" << std::endl;
    std::cout << "maximal difference is: " << at::max(l_ten_out_torch - l_ten_out) << std::endl;
  }
  else{
    std::cout << "results are the same" << std::endl;
  }  
}

int main() {
  std::cout << "running bench_binary!" << std::endl;

  blocked_matmul();

  std::cout << "finished running bench_binary!" << std::endl;
  return EXIT_SUCCESS;
}