#include "ATen/ATen.h"
#include "catch.hpp"
#include "BinaryContractionScalar.h"

TEST_CASE( "Matrix-matrix multiplication.", "[binary_contraction_scalar]" ) {
  // Test Case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

  einsum_ir::backend::BinaryContractionScalar l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   2,
                   l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );
  // data
  at::Tensor l_in_left  = at::rand( {4, 2} );
  at::Tensor l_in_right = at::rand( {3, 4} );
  at::Tensor l_out_ref  = at::rand( {3, 2} );
  at::Tensor l_out_native = l_out_ref.clone();
  at::Tensor l_out_ordered = l_out_ref.clone(); 

  // reference
  l_out_ref += at::einsum( "km,nk->nm",
                           {l_in_left, l_in_right} );

  // native input dimensions
  l_bin_cont.compile();
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_native, l_out_ref )  );

  // ordered input dimensions
  l_bin_cont.compile( einsum_ir::tenord_t::LEFT_BC_BM_BI_BK_RIGHT_BC_BN_BJ_BK_OUT_NATIVE );

  l_bin_cont.contract( l_in_left.permute( {1, 0} ).contiguous().data_ptr(),
                       l_in_right.data_ptr(),
                       l_out_ordered.data_ptr() );

  REQUIRE( at::allclose( l_out_ordered, l_out_ref )  );
}

TEST_CASE( "Binary contraction involving C, M, N and K dimensions.", "[binary_contraction_scalar]" ) {
  // Test case:
  //
  //         ______________yhgfxei________________
  //        /                                     \
  //   ygcxaei                                   yhcxfa
  //
  //   char id size type
  //      i  0    3   m0
  //      e  1    8   m1
  //      a  2    2   k0
  //      c  3    7   k1
  //      g  4    6   m2
  //      f  5    5   n0
  //      h  6    4   n1
  //      x  7    3   c0
  //      y  8    4   c1
  //
  //  ieaxcgy: 8 4 3 7 2 1 0
  //  afxchy:  8 6 3 7 5 2
  //  iexfghy: 8 6 4 5 7 1 0
  //
  //   dim types:
  //     c:  yx /  87
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 8 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 7 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4, 6 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 5, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 6, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 7, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 8, 4 ) );

  int64_t l_dim_ids_in_left[7] = { 8, 4, 3, 7, 2, 1, 0 };
  int64_t l_dim_ids_in_right[6] = { 8, 6, 3, 7, 5, 2 };
  int64_t l_dim_ids_out[7] = { 8, 6, 4, 5, 7, 1, 0 };

  einsum_ir::backend::BinaryContractionScalar l_bin_cont;
  l_bin_cont.init( 7,
                   6,
                   7,
                   l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  //                                0  1  2  3  4  5  6
  //                                y  g  c  x  a  e  i
  at::Tensor l_in_left = at::rand( {4, 6, 7, 3, 2, 8, 3} );
  //                                 0  1  2  3  4  5
  //                                 y  h  c  x  f  a
  at::Tensor l_in_right = at::rand( {4, 4, 7, 3, 5, 2} );
  //                                y  h  g  f  x  e  i
  at::Tensor l_out_ref = at::rand( {4, 4, 6, 5, 3, 8, 3} );
  at::Tensor l_out_native = l_out_ref.clone();
  at::Tensor l_out_ordered = l_out_ref.clone();

  // reference
  l_out_ref += at::einsum( "ygcxaei,yhcxfa->yhgfxei",
                           {l_in_left, l_in_right} );

  // native
  l_bin_cont.compile();
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_native, l_out_ref )  );

  // ordered
  l_bin_cont.compile( einsum_ir::tenord_t::LEFT_BC_BM_BI_BK_RIGHT_BC_BN_BJ_BK_OUT_NATIVE );

  at::Tensor l_left_ordered  = l_in_left.permute(  { 0, 3, 1, 5, 6, 2, 4 } ).contiguous();
  at::Tensor l_right_ordered = l_in_right.permute( { 0, 3, 1, 4, 2, 5 } ).contiguous();


  l_bin_cont.contract( l_left_ordered.data_ptr(),
                       l_right_ordered.data_ptr(),
                       l_out_ordered.data_ptr() );

  REQUIRE( at::allclose( l_out_ordered, l_out_ref )  );
}

TEST_CASE( "Binary contraction involving C, M, N and K dimensions using FP64, zero and relu.", "[binary_contraction_scalar]" ) {
  // Test case:
  //
  //         ______________yhgfxei________________
  //        /                                     \
  //   ygcxaei                                   yhcxfa
  //
  //   char id size type
  //      i  0    3   m0
  //      e  1    8   m1
  //      a  2    2   k0
  //      c  3    7   k1
  //      g  4    6   m2
  //      f  5    5   n0
  //      h  6    4   n1
  //      x  7    3   c0
  //      y  8    4   c1
  //
  //  ieaxcgy: 8 4 3 7 2 1 0
  //  afxchy:  8 6 3 7 5 2
  //  iexfghy: 8 6 4 5 7 1 0
  //
  //   dim types:
  //     c:  yx /  87
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 8 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 7 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4, 6 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 5, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 6, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 7, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 8, 4 ) );

  int64_t l_dim_ids_in_left[7] = { 8, 4, 3, 7, 2, 1, 0 };
  int64_t l_dim_ids_in_right[6] = { 8, 6, 3, 7, 5, 2 };
  int64_t l_dim_ids_out[7] = { 8, 6, 4, 5, 7, 1, 0 };

  einsum_ir::backend::BinaryContractionScalar l_bin_cont;
  l_bin_cont.init( 7,
                   6,
                   7,
                   l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::RELU );

  //                                0  1  2  3  4  5  6
  //                                y  g  c  x  a  e  i
  at::Tensor l_in_left = at::rand( {4, 6, 7, 3, 2, 8, 3},
                                   at::ScalarType::Double );
  //                                 0  1  2  3  4  5
  //                                 y  h  c  x  f  a
  at::Tensor l_in_right = at::rand( {4, 4, 7, 3, 5, 2},
                                    at::ScalarType::Double );
  //                                y  h  g  f  x  e  i
  at::Tensor l_out_ref = at::rand( {4, 4, 6, 5, 3, 8, 3},
                                   at::ScalarType::Double );
  at::Tensor l_out_native = l_out_ref.clone();
  at::Tensor l_out_ordered = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "ygcxaei,yhcxfa->yhgfxei",
                          {l_in_left, l_in_right} );

  l_out_ref = at::relu( l_out_ref );

  // native
  l_bin_cont.compile();
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_native, l_out_ref )  );

  // ordered
  l_bin_cont.compile( einsum_ir::tenord_t::LEFT_BC_BM_BI_BK_RIGHT_BC_BN_BJ_BK_OUT_NATIVE );

  at::Tensor l_left_ordered  = l_in_left.permute(  { 0, 3, 1, 5, 6, 2, 4 } ).contiguous();
  at::Tensor l_right_ordered = l_in_right.permute( { 0, 3, 1, 4, 2, 5 } ).contiguous();


  l_bin_cont.contract( l_left_ordered.data_ptr(),
                       l_right_ordered.data_ptr(),
                       l_out_ordered.data_ptr() );

  REQUIRE( at::allclose( l_out_ordered, l_out_ref )  );
}

TEST_CASE( "1D convolution with single input and output feature.", "[binary_contraction_scalar_conv1d]" ) {
  // Test Case:
  //
  //    ____a____
  //   /         \
  //  a           n
  //
  // char   id   size
  //    a    0      5
  //    b    1      3
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 1, 0 ) );

  int64_t l_dim_ids_left[1]  = { 0 };
  int64_t l_dim_ids_right[1] = { 1 };
  int64_t l_dim_ids_out[1]   = { 0 };

  einsum_ir::backend::BinaryContractionScalar l_bin_cont;
  l_bin_cont.init( 1,
                   1,
                   1,
                   l_dim_sizes,
                   l_dim_sizes,
                   l_dim_sizes,
                   l_dim_sizes,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );
  // data
  at::Tensor l_left  = at::rand( {1, 5+2} );
  at::Tensor l_right = at::rand( {1, 1, 3} );
  at::Tensor l_out   = at::rand( {5} );

  // reference
  at::Tensor l_out_ref = at::conv1d( l_left,
                                     l_right ).squeeze();

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "2D convolution with a single input and output feature.", "[binary_contraction_scalar_conv2d]" ) {
  // Test Case:
  //
  //    ____ab____
  //   /          \
  //  ab           cd
  //
  // char   id   size
  //    a    0     16
  //    b    1     13
  //    c    2      3
  //    d    3      3
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0,   16 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1,   13 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2,    3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3,    3 ) );

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, 16+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, 13+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3,    3 ) );

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 2, 0 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 1 ) );

  int64_t l_dim_ids_left[2]  = { 0, 1 };
  int64_t l_dim_ids_right[2] = { 2, 3 };
  int64_t l_dim_ids_out[2]   = { 0, 1 };

  einsum_ir::backend::BinaryContractionScalar l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   2,
                   l_dim_sizes_inner,
                   l_dim_sizes_outer,
                   l_dim_sizes_outer,
                   l_dim_sizes_inner,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );
  // data
  at::Tensor l_left  = at::rand( {1, 16+2, 13+2} );
  at::Tensor l_right = at::rand( {1, 1, 3, 3} );
  at::Tensor l_out   = at::rand( {16, 13} );

  // reference
  at::Tensor l_out_ref = at::conv2d( l_left,
                                     l_right ).squeeze();

  // compile
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // contract
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "2D convolution with multiple input and output features.", "[binary_contraction_scalar_conv2d]" ) {
  // Test Case:
  //
  //    ____fab____
  //   /           \
  //  eab         fecd
  //
  // char   id   size
  //    a    0     16
  //    b    1     13
  //    c    2      3
  //    d    3      3
  //    e    4      8
  //    f    5      7
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0,   16 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1,   13 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2,    3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3,    3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 4,    8 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 5,    7 ) );

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, 16+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, 13+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 4,    8 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 5,    7 ) );

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 2, 0 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 1 ) );

  int64_t l_dim_ids_left[3]  = { 4, 0, 1 };
  int64_t l_dim_ids_right[4] = { 5, 4, 2, 3 };
  int64_t l_dim_ids_out[5]   = { 5, 0, 1 };

  einsum_ir::backend::BinaryContractionScalar l_bin_cont;
  l_bin_cont.init( 3,
                   4,
                   3,
                   l_dim_sizes_inner,
                   l_dim_sizes_outer,
                   l_dim_sizes_outer,
                   l_dim_sizes_inner,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );
  // data
  at::Tensor l_left  = at::rand( {1, 8, 16+2, 13+2} );
  at::Tensor l_right = at::rand( {7, 8, 3, 3} );
  at::Tensor l_out   = at::rand( {7, 16, 13} );

  // reference
  at::Tensor l_out_ref = at::conv2d( l_left,
                                     l_right ).squeeze();

  // compile
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // contract
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "2D Convolution with multiple input and output features, weights left.", "[binary_contraction_scalar_conv2d]" ) {
  // Test Case:
  //
  //     ____fab____
  //    /           \
  //  fecd         eab
  //
  // char   id   size
  //    a    0     16
  //    b    1     13
  //    c    2      3
  //    d    3      3
  //    e    4      8
  //    f    5      7
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0,   16 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1,   13 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2,    3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3,    3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 4,    8 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 5,    7 ) );

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, 16+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, 13+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 4,    8 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 5,    7 ) );

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 2, 0 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 1 ) );

  int64_t l_dim_ids_left[4]  = { 5, 4, 2, 3 };
  int64_t l_dim_ids_right[3] = { 4, 0, 1 };
  int64_t l_dim_ids_out[5]   = { 5, 0, 1 };

  einsum_ir::backend::BinaryContractionScalar l_bin_cont;
  l_bin_cont.init( 4,
                   3,
                   3,
                   l_dim_sizes_inner,
                   l_dim_sizes_outer,
                   l_dim_sizes_outer,
                   l_dim_sizes_inner,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );
  // data
  at::Tensor l_left  = at::rand( {7, 8, 3, 3} );
  at::Tensor l_right = at::rand( {1, 8, 16+2, 13+2} );
  at::Tensor l_out   = at::rand( {7, 16, 13} );

  // reference
  at::Tensor l_out_ref = at::conv2d( l_right,
                                     l_left ).squeeze();

  // compile
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // contract
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "3D Convolution with single input and output feature.", "[binary_contraction_scalar_conv3d]" ) {
  // Test Case:
  //
  //    ____abc____
  //   /           \
  // abc           def
  //
  // char   id   size
  //    a    0     16
  //    b    1     13
  //    c    2      7
  //    d    3      3
  //    e    4      3
  //    f    5      3
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0,   16 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1,   13 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2,    7 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3,    5 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 4,    4 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 5,    3 ) );

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0,   16+4 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1,   13+3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2,    7+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3,      5 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 4,      4 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 5,      3 ) );

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 0 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 4, 1 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 5, 2 ) );

  int64_t l_dim_ids_left[3]  = { 0, 1, 2 };
  int64_t l_dim_ids_right[3] = { 3, 4, 5 };
  int64_t l_dim_ids_out[3]   = { 0, 1, 2 };

  einsum_ir::backend::BinaryContractionScalar l_bin_cont;
  l_bin_cont.init( 3,
                   3,
                   3,
                   l_dim_sizes_inner,
                   l_dim_sizes_outer,
                   l_dim_sizes_outer,
                   l_dim_sizes_inner,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );
  // data
  at::Tensor l_left  = at::rand( {1, 16+4, 13+3, 7+2} );
  at::Tensor l_right = at::rand( {1, 1, 5, 4, 3} );
  at::Tensor l_out   = at::rand( {16, 13, 7} );

  // reference
  at::Tensor l_out_ref = at::conv3d( l_left,
                                     l_right ).squeeze();

  // compile
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // contract
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}