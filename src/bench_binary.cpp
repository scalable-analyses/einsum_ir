#include <cstdlib>
#include <iostream>

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
  double l_time_compile = 0;
  double l_time = 0;
  double l_gflops = 0;

  // ./gemm_kernel F32 F32 F32 F32 64 8 24 64 24 64 1 0 0 0 0 0 0 0 0 nopf nobr 1 0 1000000 0
  //
  // C: 256
  // M: 512
  // N: 128
  // K: 768
  int64_t l_size_c = 256;
  int64_t l_size_m = 512;
  int64_t l_size_n = 128;
  int64_t l_size_k = 768;

  int64_t l_size_c0 = 4;
  int64_t l_size_c1 = 8;
  int64_t l_size_c2 = 8;

  int64_t l_size_m0 = 8;
  int64_t l_size_m1 = 64;

  int64_t l_size_n0 = 16;
  int64_t l_size_n1 = 8;

  int64_t l_size_k0 = 2;
  int64_t l_size_k1 = 16;
  int64_t l_size_k2 = 24;

  l_n_flops = l_size_c * l_size_m * l_size_n * l_size_k * 2;

  /**
   * Matmul 
   **/
  at::Tensor l_mat_a = at::rand( { l_size_c, l_size_k, l_size_m } );
  at::Tensor l_mat_b = at::rand( { l_size_c, l_size_n, l_size_k } );
  at::Tensor l_mat_c = at::rand( { l_size_c, l_size_n, l_size_m } );

  std::cout << "matmul:" << std::endl;

  // dry run
  at::Tensor l_out_matmul = at::matmul( l_mat_b, l_mat_a );

  l_tp0 = std::chrono::steady_clock::now();
  l_out_matmul = at::matmul( l_mat_b, l_mat_a );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_n_flops / l_time;

  std::cout << "  time:   " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;

  /**
   * einsum data
   **/
  at::Tensor l_ten_left = l_mat_a;
  l_ten_left = l_ten_left.view( { l_size_c0,     // 0
                                  l_size_c1,     // 1
                                  l_size_c2,     // 2
                                  l_size_k0,     // 3
                                  l_size_k1,     // 4
                                  l_size_k2,     // 5
                                  l_size_m0,     // 6
                                  l_size_m1 } ); // 7
  //                                c0 c1 c2 m0 k0 k1 k2 m1
  l_ten_left = l_ten_left.permute( { 0, 1, 2, 6, 3, 4, 5, 7} ).contiguous();

  at::Tensor l_ten_right = l_mat_b;
  l_ten_right = l_ten_right.view( { l_size_c0,     // 0
                                    l_size_c1,     // 1
                                    l_size_c2,     // 2
                                    l_size_n0,     // 3
                                    l_size_n1,     // 4
                                    l_size_k0,     // 5
                                    l_size_k1,     // 6
                                    l_size_k2 } ); // 7
  //                                  c0 c1 c2 n0 k0 k1 n1 k2
  l_ten_right = l_ten_right.permute( { 0, 1, 2, 3, 5, 6, 4, 7} ).contiguous();

  at::Tensor l_ten_out = l_mat_c;
  l_ten_out = l_ten_out.view( { l_size_c0,     // 0
                                l_size_c1,     // 1
                                l_size_c2,     // 2
                                l_size_m0,     // 3
                                l_size_m1,     // 4
                                l_size_n0,     // 5
                                l_size_n1 } ); // 6
  //                              c0 c1 c2 n0 m0 n1 m1
  l_ten_out = l_ten_out.permute( { 0, 1, 2, 5, 3, 6, 4} ).contiguous();

  /*
   * at::einsum
   */
  std::cout << "at::einsum:" << std::endl;
  // c: abc
  // m: de
  // n: fg
  // k: hij

  // dry run
  at::Tensor l_ten_out_torch = at::einsum( "abcdhije,abcfhigj->abcfdge",
                                            {l_ten_left, l_ten_right},
                                            { {0,1} } );

  l_tp0 = std::chrono::steady_clock::now();
  l_ten_out_torch = at::einsum( "abcdhije,abcfhigj->abcfdge",
                                {l_ten_left, l_ten_right},
                                { {0,1} } );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_n_flops / l_time;

  std::cout << "  time (contract): " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;

  /*
   * einsum_ir
   */
  std::cout << "einsum_ir:" << std::endl;
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, l_size_c0 ) ); // c0
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, l_size_c1 ) ); // c1
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, l_size_c2 ) ); // c2

  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, l_size_m0 ) ); // m0
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4, l_size_m1 ) ); // m1

  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 5, l_size_n0 ) ); // n0
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 6, l_size_n1 ) ); // n1

  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 7, l_size_k0 ) ); // k0
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 8, l_size_k1 ) ); // k1
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 9, l_size_k2 ) ); // k2

  //                               c0 c1 c2 m0 k0 k1 k2 m1
  int64_t l_dim_ids_in_left[8]  = { 0, 1, 2, 3, 7, 8, 9, 4 };
  //                               c0 c1 c2 n0 k0 k1 n1 k2
  int64_t l_dim_ids_in_right[8] = { 0, 1, 2, 5, 7, 8, 6, 9 };
  //                               c0 c1 c2 n0 m0 n1 m1
  int64_t l_dim_ids_out[7]      = { 0, 1, 2, 5, 3, 6, 4 };

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 8,
                   8,
                   7,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  l_tp0 = std::chrono::steady_clock::now();
  l_bin_cont.compile();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  // enable threading
#ifdef _OPENMP
  // four times overload
  int64_t l_num_tasks = omp_get_max_threads() * 4;

  l_bin_cont.threading( l_num_tasks );
#endif

  at::Tensor l_ten_left_perm = l_ten_left;
  at::Tensor l_ten_right_perm = l_ten_right.permute( {0, 1, 2, 3, 4, 6, 5, 7} ).contiguous();

  // dry run
  l_bin_cont.contract( l_ten_left_perm.data_ptr(),
                       l_ten_right_perm.data_ptr(),
                       l_ten_out.data_ptr() );

  l_tp0 = std::chrono::steady_clock::now();
  l_bin_cont.contract( l_ten_left_perm.data_ptr(),
                       l_ten_right_perm.data_ptr(),
                       l_ten_out.data_ptr() );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_n_flops / l_time;

  std::cout << "  time (compile): " << l_time_compile << std::endl;
  std::cout << "  time (contract): " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;

  if( !at::allclose( l_ten_out_torch, l_ten_out ) ) {
    std::cerr << "error: einsum_ir solution is not close to aten!" << std::endl;
  }
}

int main() {
  std::cout << "running bench_binary!" << std::endl;

  blocked_matmul();

  std::cout << "finished running bench_binary!" << std::endl;
  return EXIT_SUCCESS;
}