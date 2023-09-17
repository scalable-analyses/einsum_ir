#include <cstdlib>
#include <iostream>

#include <ATen/ATen.h>
#include "backend/BinaryContractionTpp.h"

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

  at::Tensor l_out_matmul = l_mat_c.clone();

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
  at::Tensor l_ten_out_torch = l_ten_out.clone();

  /*
   * at::einsum
   */
  std::cout << "at::einsum:" << std::endl;
  // c: abc
  // m: de
  // n: fg
  // k: hij
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
                   nullptr,
                   nullptr,
                   nullptr,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   nullptr,
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

  l_bin_cont.threading( 256 );

  at::Tensor l_ten_left_perm = l_ten_left;
  at::Tensor l_ten_right_perm = l_ten_right.permute( {0, 1, 2, 3, 4, 6, 5, 7} ).contiguous();

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

void conv2d() {
  std::cout << "***********************" << std::endl;
  std::cout << "*** conv2d testcase ***" << std::endl;
  std::cout << "***********************" << std::endl;

  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_num_flops = 0;
  double l_time_compile = 0;
  double l_time = 0;
  double l_gflops = 0;

  // Test Case:
  //
  //     ____fab____
  //    /           \
  //  eab          fecd
  //
  // char   id   size
  //    a    0     28 // height
  //    b    1     28 // width
  //    c    2      3 // 2nd conv dim
  //    d    3      3 // 1st conv dim
  //    e    4    128 // in features
  //    f    5    128 // out features
  int64_t l_width         = 28;
  int64_t l_height        = 28;
  int64_t l_kernel_size_0 = 3;
  int64_t l_kernel_size_1 = 3;
  int64_t l_features_in   = 128;
  int64_t l_features_out  = 128;

  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0, l_height        ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1, l_width         ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2, l_kernel_size_1 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3, l_kernel_size_0 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 4, l_features_in   ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 5, l_features_out  ) );

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, l_height+2      ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, l_width+2       ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2, l_kernel_size_1 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3, l_kernel_size_0 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 4, l_features_in   ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 5, l_features_out  ) );

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 2, 0 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 1 ) );

  int64_t l_dim_ids_left[4]  = { 5, 4, 2, 3 };
  int64_t l_dim_ids_right[3] = { 4, 0, 1 };
  int64_t l_dim_ids_out[3]   = { 5, 0, 1 };

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 4,
                   3,
                   3,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_outer,
                   &l_dim_sizes_outer,
                   nullptr,
                   &l_dim_sizes_inner,
                   nullptr,
                   nullptr,
                   nullptr,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   &l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::COPY,
                   einsum_ir::MADD,
                   einsum_ir::RELU );

  // compile
  l_tp0 = std::chrono::steady_clock::now();
  einsum_ir::err_t l_err = l_bin_cont.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    std::cerr << "error: failed to compile the convolution!" << std::endl;
    return;
  }
  l_bin_cont.threading( 8 );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time_compile = l_dur.count();

  l_num_flops = l_bin_cont.num_ops();

  // data
  at::Tensor l_left  = at::randn( { l_features_out,
                                    l_features_in,
                                    l_kernel_size_1,
                                    l_kernel_size_0 } );
  at::Tensor l_right = at::randn( { 1,
                                    l_features_in,
                                    l_height+2,
                                    l_width+2 } );
  at::Tensor l_bias  = at::randn( { l_features_out,
                                    l_height,
                                    l_width } );
  at::Tensor l_out   = at::randn( { l_features_out,
                                    l_height,
                                    l_width } );

  /*
   * at::conv2d
   */
  std::cout << "at::conv2d:" << std::endl;

  at::Tensor l_out_ref = at::relu( l_bias + at::conv2d( l_right,
                                                        l_left ) ).squeeze();

  l_tp0 = std::chrono::steady_clock::now();
  l_out_ref = at::relu( l_bias + at::conv2d( l_right,
                                             l_left ) ).squeeze();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  time:   " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;

  at::Tensor l_left_perm  = l_left.permute( {2, 3, 0, 1} ).contiguous();
  at::Tensor l_right_perm = l_right.permute( {0, 2, 1, 3} ).contiguous();

  /*
   * einsum_ir
   */
  std::cout << "einsum_ir" << std::endl;

  l_tp0 = std::chrono::steady_clock::now();
  l_bin_cont.contract( l_left_perm.data_ptr(),
                       l_right_perm.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  time (compile): " << l_time_compile << std::endl;
  std::cout << "  time (eval):    " << l_time << std::endl;
  std::cout << "  gflops:         " << l_gflops << std::endl;

  if( !at::allclose( l_out, l_out_ref, 1E-4, 1E-4 ) ) {
    std::cerr << "error: solution is not close!" << std::endl;
  }
}

int main() {
  std::cout << "running bench_binary!" << std::endl;

  blocked_matmul();
  conv2d();

  std::cout << "finished running bench_binary!" << std::endl;
  return EXIT_SUCCESS;
}