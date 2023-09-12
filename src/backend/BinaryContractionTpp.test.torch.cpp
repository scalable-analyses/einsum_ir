#include "ATen/ATen.h"
#include "catch.hpp"
#include "BinaryContractionTpp.h"

TEST_CASE( "TPP-based binary contraction executing matmuls.", "[bin_cont_tpp]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

  // data layout
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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

  // reference
  l_out_ref += at::einsum( "km,nk->nm",
                           {l_in_left, l_in_right} );

  // compile contraction
  l_bin_cont.compile();

  // execute
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_ref, l_out_native )  );
}

TEST_CASE( "TPP-based binary contraction executing a batched matmul.", "[bin_cont_tpp]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 5 ) );

  int64_t l_dim_ids_in_left[3]  = { 3, 1, 0 };
  int64_t l_dim_ids_in_right[3] = { 2, 3, 0 };
  int64_t l_dim_ids_out[3]      = { 2, 1, 0 };

  // data layout
  //
  //    ____nmc___
  //   /          \
  // kmc           nkc
  //
  // char   id   size
  //    c    0      2
  //    m    1      3
  //    n    2      4
  //    k    3      5
  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 3,
                   3,
                   3,
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
  at::Tensor l_in_left  = at::rand( {5, 3, 2} );
  at::Tensor l_in_right = at::rand( {4, 5, 2} );
  at::Tensor l_out_ref  = at::rand( {4, 3, 2} );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref += at::einsum( "kmc,nkc->nmc",
                           {l_in_left, l_in_right} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_ref, l_out_native )  );
}

TEST_CASE( "TPP-based binary contraction executing matmuls with FP64 and zero first touch.", "[bin_cont_tpp]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

  // data layout
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   2,
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
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_in_left  = at::rand( {4, 2},
                                    at::ScalarType::Double );
  at::Tensor l_in_right = at::rand( {3, 4},
                                    at::ScalarType::Double );
  at::Tensor l_out_ref  = at::rand( {3, 2},
                                    at::ScalarType::Double );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "km,nk->nm",
                          {l_in_left, l_in_right} );

  // compile contraction
  l_bin_cont.compile();

  // execute
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_ref, l_out_native )  );
}

TEST_CASE( "TPP-based Binary contraction involving C, M, N and K dimensions, stride-1 M.", "[bin_cont_tpp]" ) {
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
  //  yhgfxei: 8 4 3 7 2 1 0
  //  yhcxfa:  8 6 3 7 5 2
  //  ygcxaei: 8 6 4 5 7 1 0
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

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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

  l_bin_cont.compile();

  // TPP-kernel will use blocking:
  //   mb: e, i
  //   nb: f
  //   kb: c, a
  // ordering:
  //   left  (BC-BM-BK-KB-MB): yx - g - - ca - ei
  //   right (BC-BN-BK-NB-KB): yx - h - - f  - ca
  at::Tensor l_left_ordered  = l_in_left.permute(  { 0, 3, 1, 2, 4, 5, 6 } ).contiguous();
  at::Tensor l_right_ordered = l_in_right.permute( { 0, 3, 1, 4, 2, 5 } ).contiguous();


  l_bin_cont.contract( l_left_ordered.data_ptr(),
                       l_right_ordered.data_ptr(),
                       l_out_ordered.data_ptr() );

  REQUIRE( at::allclose( l_out_ordered, l_out_ref )  );
}

TEST_CASE( "TPP-based Binary contraction involving C, M, N and K dimensions, stride-1 M, FP64, zero first touch, relu last touch.", "[bin_cont_tpp]" ) {
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

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::RELU );

  //                                0  1  2  3  4  5  6
  //                                y  g  c  x  a  e  i
  at::Tensor l_in_left = at::rand( {4, 6, 7, 3, 2, 8, 3},
                                   at::ScalarType::Float );
  //                                 0  1  2  3  4  5
  //                                 y  h  c  x  f  a
  at::Tensor l_in_right = at::rand( {4, 4, 7, 3, 5, 2},
                                    at::ScalarType::Float );
  //                                y  h  g  f  x  e  i
  at::Tensor l_out_ref = at::rand( {4, 4, 6, 5, 3, 8, 3},
                                   at::ScalarType::Float );
  at::Tensor l_out_native = l_out_ref.clone();
  at::Tensor l_out_ordered = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "ygcxaei,yhcxfa->yhgfxei",
                          {l_in_left, l_in_right} );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // TPP-kernel will use blocking:
  //   mb: e, i
  //   nb: f
  //   kb: c, a
  // ordering:
  //   left  (BC-BM-BK-KB-MB): yx - g - - ca - ei
  //   right (BC-BN-BK-NB-KB): yx - h - - f  - ca
  at::Tensor l_left_ordered  = l_in_left.permute(  { 0, 3, 1, 2, 4, 5, 6 } ).contiguous();
  at::Tensor l_right_ordered = l_in_right.permute( { 0, 3, 1, 4, 2, 5 } ).contiguous();


  l_bin_cont.contract( l_left_ordered.data_ptr(),
                       l_right_ordered.data_ptr(),
                       l_out_ordered.data_ptr() );

  REQUIRE( at::allclose( l_out_ordered, l_out_ref )  );
}

TEST_CASE( "TPP-based Binary contraction involving C, M, N and K dimensions, stride-1 C.", "[bin_cont_tpp]" ) {
  // Test case:
  //
  //         ______________hgfeixy________________
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
  int64_t l_dim_ids_out[7] = { 6, 4, 5, 1, 0, 7, 8 };

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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
  //                                h  g  f  e  i  x  y
  at::Tensor l_out_ref = at::rand( {4, 6, 5, 8, 3, 3, 4} );
  at::Tensor l_out_native = l_out_ref.clone();
  at::Tensor l_out_ordered = l_out_ref.clone();

  // reference
  l_out_ref += at::einsum( "ygcxaei,yhcxfa->hgfeixy",
                           {l_in_left, l_in_right} );

  l_bin_cont.compile();
  at::Tensor l_left_ordered  = l_in_left.permute(  { 1, 5, 2, 4, 6, 3, 0 } ).contiguous();
  at::Tensor l_right_ordered = l_in_right.permute( { 1, 2, 4, 5, 3, 0 } ).contiguous();


  l_bin_cont.contract( l_left_ordered.data_ptr(),
                       l_right_ordered.data_ptr(),
                       l_out_ordered.data_ptr() );

  REQUIRE( at::allclose( l_out_ordered, l_out_ref )  );
}

TEST_CASE( "TPP-based 1D Convolution with a single input feature.", "[bin_cont_tpp_conv]" ) {
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

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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

TEST_CASE( "TPP-based 1D Convolution with additional K.", "[bin_cont_tpp_conv]" ) {
  // Test Case:
  //
  //    ____a____
  //   /         \
  // ca          cb
  //
  // char   id   size
  //    a    0      5
  //    b    1      3
  //    c    2      8
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0, 5 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2, 8 ) );

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, 5+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2, 8 ) );


  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 1, 0 ) );

  int64_t l_dim_ids_left[2]  = { 2, 0 };
  int64_t l_dim_ids_right[2] = { 2, 1 };
  int64_t l_dim_ids_out[1]   = { 0 };

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   1,
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
  at::Tensor l_left  = at::rand( {8, 5+2} );
  at::Tensor l_right = at::ones( {1, 8, 3} );
  at::Tensor l_out   = at::ones( {5} );

  // reference
  at::Tensor l_out_ref = at::conv1d( l_left,
                                     l_right ).squeeze();

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  at::Tensor l_left_perm = l_left;
  at::Tensor l_right_perm = l_right.permute( {0, 2, 1} ).contiguous();

  l_bin_cont.contract( l_left_perm.data_ptr(),
                       l_right_perm.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "TPP-based 2D Convolution with a single input feature.", "[bin_cont_tpp_conv]" ) {
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

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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

TEST_CASE( "TPP-based 2D Convolution with additional K and ReLU, weights right.", "[bin_cont_tpp_conv]" ) {
  // Test Case:
  //
  //    ____ab____
  //   /          \
  //  eab         ecd
  //
  // char   id   size
  //    a    0     11
  //    b    1     16
  //    c    2      3
  //    d    3      3
  //    e    4      6
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0,  11 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1,  16 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2,   3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3,   3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 4,   6 ) );

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, 11+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, 16+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 4,    6 ) );


  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 2, 0 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 1 ) );

  int64_t l_dim_ids_left[3]  = { 4, 0, 1 };
  int64_t l_dim_ids_right[3] = { 4, 2, 3 };
  int64_t l_dim_ids_out[2]   = { 0, 1 };

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 3,
                   3,
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
                   einsum_ir::RELU );
  // data
  at::Tensor l_left  = at::randn( {1, 6, 11+2, 16+2} );
  at::Tensor l_right = at::randn( {1, 6, 3, 3} );
  at::Tensor l_out   = at::rand( {11, 16} );

  // reference
  at::Tensor l_out_ref = at::conv2d( l_left,
                                     l_right ).squeeze();
  l_out_ref = at::relu( l_out_ref );

  // compile
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  at::Tensor l_left_perm  = l_left.permute(  {0, 2, 1, 3} ).contiguous();
  at::Tensor l_right_perm = l_right.permute( {0, 2, 3, 1} ).contiguous();

  // contract
  l_bin_cont.contract( l_left_perm.data_ptr(),
                       l_right_perm.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-7 )  );
}

TEST_CASE( "TPP-based 2D Convolution with additional K and ReLU, weights left.", "[bin_cont_tpp_conv]" ) {
  // Test Case:
  //
  //    ____ab____
  //   /          \
  //  eab         ecd
  //
  // char   id   size
  //    a    0     11
  //    b    1     16
  //    c    2      3
  //    d    3      3
  //    e    4      6
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 0,  11 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 1,  16 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 2,   3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 3,   3 ) );
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 4,   6 ) );

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 0, 11+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 1, 16+2 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 2,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 3,    3 ) );
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 4,    6 ) );


  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 2, 0 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 3, 1 ) );

  int64_t l_dim_ids_left[3]  = { 4, 2, 3 };
  int64_t l_dim_ids_right[3] = { 4, 0, 1 };
  int64_t l_dim_ids_out[2]   = { 0, 1 };

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 3,
                   3,
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
                   einsum_ir::RELU );
  // data
  at::Tensor l_left  = at::randn( {1, 6, 3, 3} );
  at::Tensor l_right = at::randn( {1, 6, 11+2, 16+2} );
  at::Tensor l_out   = at::randn( {11, 16} );

  // reference
  at::Tensor l_out_ref = at::conv2d( l_right,
                                     l_left ).squeeze();
  l_out_ref = at::relu( l_out_ref );

  // compile
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  at::Tensor l_left_perm  = l_left.permute(  {0, 2, 3, 1} ).contiguous();
  at::Tensor l_right_perm = l_right.permute( {0, 2, 1, 3} ).contiguous();

  // contract
  l_bin_cont.contract( l_left_perm.data_ptr(),
                       l_right_perm.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-7 )  );
}

TEST_CASE( "TPP-based 2D Convolution with input and output features, weights left.", "[bin_cont_tpp_conv]" ) {
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
  int64_t l_dim_ids_out[3]   = { 5, 0, 1 };

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_left  = at::randn( {7, 8, 3, 3},
                                  at::ScalarType::Double );
  at::Tensor l_right = at::randn( {1, 8, 16+2, 13+2},
                                  at::ScalarType::Double );
  at::Tensor l_out   = at::randn(  {7, 16, 13},
                                   at::ScalarType::Double );

  // reference
  at::Tensor l_out_ref = at::conv2d( l_right,
                                     l_left ).squeeze();

  // compile
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  at::Tensor l_left_perm  = l_left.permute( {2, 3, 0, 1} ).contiguous();
  at::Tensor l_right_perm = l_right.permute( {0, 2, 1, 3} ).contiguous();

  // contract
  l_bin_cont.contract( l_left_perm.data_ptr(),
                       l_right_perm.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-7 )  );
}