#include "ATen/ATen.h"
#include "catch.hpp"
#include "BinaryContractionBlas.h"

TEST_CASE( "FP32 BLAS-based binary contraction executing a matmul.", "[binary_contraction_blas]" ) {
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
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   2,
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
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_in_left  = at::randn( {4, 2} );
  at::Tensor l_in_right = at::randn( {3, 4} );
  at::Tensor l_out_ref  = at::randn( {3, 2} );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref += at::einsum( "km,nk->nm",
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

TEST_CASE( "Complex FP32 BLAS-based binary contraction executing a matmul.", "[binary_contraction_blas]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 4 ) );

  int64_t l_dim_ids_in_left[3]  = { 0, 3, 1 };
  int64_t l_dim_ids_in_right[3] = { 0, 2, 3 };
  int64_t l_dim_ids_out[3]      = { 0, 2, 1 };

  // data layout
  //
  //     ____cnm___
  //    /          \
  // ckm           cnk
  //
  // char   id   size
  //    c    0      2 // complex
  //    m    1      2
  //    n    2      3
  //    k    3      4
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 3,
                   3,
                   3,
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
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_left  = at::randn( {2, 4, 2},
                                  at::ScalarType::Float );
  at::Tensor l_right = at::randn( {2, 3, 4},
                                  at::ScalarType::Float );
  at::Tensor l_out   = at::randn( {2, 3, 2},
                                  at::ScalarType::Float );

  // reference
  at::Tensor l_left_aos    = at::view_as_complex( l_left.permute(  { 1, 2, 0 } ).contiguous() );
  at::Tensor l_right_aos   = at::view_as_complex( l_right.permute( { 1, 2, 0 } ).contiguous() );
  at::Tensor l_out_ref_aos = at::view_as_complex( l_out.permute(   { 1, 2, 0 } ).contiguous() );

  l_out_ref_aos += at::einsum( "km,nk->nm",
                               {l_left_aos, l_right_aos} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( { 1, 2, 0 } ).contiguous() );

  REQUIRE( at::allclose( l_out_aos, l_out_ref_aos )  );
}

TEST_CASE( "Complex FP32 BLAS-based binary contraction executing a matmul with zeroing.", "[binary_contraction_blas]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 4 ) );

  int64_t l_dim_ids_in_left[3]  = { 0, 3, 1 };
  int64_t l_dim_ids_in_right[3] = { 0, 2, 3 };
  int64_t l_dim_ids_out[3]      = { 0, 2, 1 };

  // data layout
  //
  //     ____cnm___
  //    /          \
  // ckm           cnk
  //
  // char   id   size
  //    c    0      2 // complex
  //    m    1      2
  //    n    2      3
  //    k    3      4
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 3,
                   3,
                   3,
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
                   einsum_ir::CPX_ZERO,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_left  = at::randn( {2, 4, 2},
                                  at::ScalarType::Float );
  at::Tensor l_right = at::randn( {2, 3, 4},
                                  at::ScalarType::Float );
  at::Tensor l_out   = at::randn( {2, 3, 2},
                                  at::ScalarType::Float );

  // reference
  at::Tensor l_left_aos    = at::view_as_complex( l_left.permute(  { 1, 2, 0 } ).contiguous() );
  at::Tensor l_right_aos   = at::view_as_complex( l_right.permute( { 1, 2, 0 } ).contiguous() );
  at::Tensor l_out_ref_aos = at::view_as_complex( l_out.permute(   { 1, 2, 0 } ).contiguous() );

  l_out_ref_aos = at::einsum( "km,nk->nm",
                              {l_left_aos, l_right_aos} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( { 1, 2, 0 } ).contiguous() );

  REQUIRE( at::allclose( l_out_aos, l_out_ref_aos )  );
}

TEST_CASE( "Complex FP64 BLAS-based binary contraction executing a matmul.", "[binary_contraction_blas]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 4 ) );

  int64_t l_dim_ids_in_left[3]  = { 0, 3, 1 };
  int64_t l_dim_ids_in_right[3] = { 0, 2, 3 };
  int64_t l_dim_ids_out[3]      = { 0, 2, 1 };

  // data layout
  //
  //     ____cnm___
  //    /          \
  // ckm           cnk
  //
  // char   id   size
  //    c    0      2 // complex
  //    m    1      2
  //    n    2      3
  //    k    3      4
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 3,
                   3,
                   3,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_left  = at::randn( {2, 4, 2},
                                  at::ScalarType::Double );
  at::Tensor l_right = at::randn( {2, 3, 4},
                                  at::ScalarType::Double );
  at::Tensor l_out   = at::randn( {2, 3, 2},
                                  at::ScalarType::Double );

  // reference
  at::Tensor l_left_aos    = at::view_as_complex( l_left.permute(  { 1, 2, 0 } ).contiguous() );
  at::Tensor l_right_aos   = at::view_as_complex( l_right.permute( { 1, 2, 0 } ).contiguous() );
  at::Tensor l_out_ref_aos = at::view_as_complex( l_out.permute(   { 1, 2, 0 } ).contiguous() );

  l_out_ref_aos += at::einsum( "km,nk->nm",
                               {l_left_aos, l_right_aos} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( { 1, 2, 0 } ).contiguous() );

  REQUIRE( at::allclose( l_out_aos, l_out_ref_aos )  );
}

TEST_CASE( "Complex FP64 BLAS-based binary contraction executing a matmul with zeroing.", "[binary_contraction_blas]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 4 ) );

  int64_t l_dim_ids_in_left[3]  = { 0, 3, 1 };
  int64_t l_dim_ids_in_right[3] = { 0, 2, 3 };
  int64_t l_dim_ids_out[3]      = { 0, 2, 1 };

  // data layout
  //
  //     ____cnm___
  //    /          \
  // ckm           cnk
  //
  // char   id   size
  //    c    0      2 // complex
  //    m    1      2
  //    n    2      3
  //    k    3      4
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 3,
                   3,
                   3,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::CPX_ZERO,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_left  = at::randn( {2, 4, 2},
                                  at::ScalarType::Double );
  at::Tensor l_right = at::randn( {2, 3, 4},
                                  at::ScalarType::Double );
  at::Tensor l_out   = at::randn( {2, 3, 2},
                                  at::ScalarType::Double );

  // reference
  at::Tensor l_left_aos    = at::view_as_complex( l_left.permute(  { 1, 2, 0 } ).contiguous() );
  at::Tensor l_right_aos   = at::view_as_complex( l_right.permute( { 1, 2, 0 } ).contiguous() );
  at::Tensor l_out_ref_aos = at::view_as_complex( l_out.permute(   { 1, 2, 0 } ).contiguous() );

  l_out_ref_aos = at::einsum( "km,nk->nm",
                              {l_left_aos, l_right_aos} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( { 1, 2, 0 } ).contiguous() );

  REQUIRE( at::allclose( l_out_aos, l_out_ref_aos )  );
}

TEST_CASE( "FP64 BLAS-based binary contraction executing a packed matmul.", "[binary_contraction_blas]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 3 ) );

  int64_t l_dim_ids_in_left[3]  = { 3, 2, 0 };
  int64_t l_dim_ids_in_right[3] = { 3, 1, 2 };
  int64_t l_dim_ids_out[3]      = { 1, 0, 3 };

  // data layout
  //
  //    ____nmc___
  //   /          \
  // ckm          cnk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  //    c    3      3
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 3,
                   3,
                   3,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_left    = at::randn( {3, 4, 2},
                                    at::ScalarType::Double );
  at::Tensor l_right   = at::randn( {3, 3, 4},
                                    at::ScalarType::Double );
  at::Tensor l_out_ref = at::randn( {3, 2, 3},
                                    at::ScalarType::Double );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref += at::einsum( "ckm,cnk->nmc",
                           {l_left, l_right} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_ref, l_out_native )  );
}

TEST_CASE( "Complex FP64 BLAS-based binary contraction executing a packed matmul.", "[binary_contraction_blas]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4, 3 ) );

  int64_t l_dim_ids_in_left[4]  = { 0, 4, 3, 1 };
  int64_t l_dim_ids_in_right[4] = { 0, 4, 2, 3 };
  int64_t l_dim_ids_out[4]      = { 0, 2, 1, 4 };

  // data layout
  //
  //     ____dnmc___
  //    /           \
  // dckm           dcnk
  //
  // char   id   size
  //    d    0      2 // complex
  //    m    1      2
  //    n    2      3
  //    k    3      4
  //    c    4      3
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 4,
                   4,
                   4,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_left    = at::randn( {2, 3, 4, 2},
                                    at::ScalarType::Double );
  at::Tensor l_right   = at::randn( {2, 3, 3, 4},
                                    at::ScalarType::Double );
  at::Tensor l_out     = at::randn( {2, 3, 2, 3},
                                    at::ScalarType::Double );

  at::Tensor l_left_aos    = at::view_as_complex( l_left.permute(  { 1, 2, 3, 0 } ).contiguous() );
  at::Tensor l_right_aos   = at::view_as_complex( l_right.permute( { 1, 2, 3, 0 } ).contiguous() );
  at::Tensor l_out_ref_aos = at::view_as_complex( l_out.permute(   { 1, 2, 3, 0 } ).contiguous() );

  // reference
  l_out_ref_aos += at::einsum( "ckm,cnk->nmc",
                               {l_left_aos, l_right_aos} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );


  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( { 1, 2, 3, 0 } ).contiguous() );
  REQUIRE( at::allclose( l_out_ref_aos, l_out_aos )  );
}

TEST_CASE( "FP32 BLAS-based binary contraction involving C, M, N and K dimensions, stride-1 M.", "[binary_contraction_blas]" ) {
  // Test case:
  //
  //         ______________yhgfxei________________
  //        /                                     \
  //   yxgcaei                                   yxhfca
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
  //  yhgfxei: 8 6 4 5 7 1 0
  //  yxgcaei: 8 7 4 3 2 1 0
  //  yxhfca:  8 7 6 5 3 2
  //
  //   dim types:
  //     c:  yx /  87
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32
  //
  // BLAS call will use blocking:
  //   mb: e, i
  //   nb: f
  //   kb: c, a
  // ordering:
  //   left  (BC-BM-BK-KB-MB): yx - g - - ca - ei
  //   right (BC-BN-BK-NB-KB): yx - h - - f  - ca

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

  int64_t l_dim_ids_out[7] = { 8, 6, 4, 5, 7, 1, 0 };
  int64_t l_dim_ids_left[7] = { 8, 7, 4, 3, 2, 1, 0 };
  int64_t l_dim_ids_right[6] = { 8, 7, 6, 5, 3, 2 };

  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 7,
                   6,
                   7,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  //                              y  x  g  c  a  e  i
  at::Tensor l_left = at::randn( {4, 3, 6, 7, 2, 8, 3} );
  //                               y  x  h  f  c  a
  at::Tensor l_right = at::randn( {4, 3, 4, 5, 7, 2} );
  //                                y  h  g  f  x  e  i
  at::Tensor l_out_ref = at::randn( {4, 4, 6, 5, 3, 8, 3} );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "yxgcaei,yxhfca->yhgfxei",
                          {l_left, l_right} );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_native, l_out_ref, 1E-4, 1E-6 )  );
}

TEST_CASE( "Complex FP32 BLAS-based binary contraction involving C, M, N and K dimensions, stride-1 M.", "[binary_contraction_blas]" ) {
  // Test case:
  //
  //         ______________yhgfxei________________
  //        /                                     \
  //   yxgcaei                                   yxhfca
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
  //      z  9    2   c2
  //
  //  zyhgfxei: 9 8 6 4 5 7 1 0
  //  zyxgcaei: 9 8 7 4 3 2 1 0
  //  zyxhfca:  9 8 7 6 5 3 2
  //
  //   dim types:
  //     c: zyx / 987
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32
  //
  // BLAS call will use blocking:
  //   mb: e, i
  //   nb: f
  //   kb: c, a
  // ordering:
  //   left  (BC-BM-BK-KB-MB): yx - g - - ca - ei
  //   right (BC-BN-BK-NB-KB): yx - h - - f  - ca

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
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 9, 2 ) );

  int64_t l_dim_ids_out[8] = { 9, 8, 6, 4, 5, 7, 1, 0 };
  int64_t l_dim_ids_left[8] = { 9, 8, 7, 4, 3, 2, 1, 0 };
  int64_t l_dim_ids_right[7] = { 9, 8, 7, 6, 5, 3, 2 };

  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 8,
                   7,
                   8,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::CPX_ZERO,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  //                              z  y  x  g  c  a  e  i
  at::Tensor l_left = at::randn( {2, 4, 3, 6, 7, 2, 8, 3} );
  //                               z  y  x  h  f  c  a
  at::Tensor l_right = at::randn( {2, 4, 3, 4, 5, 7, 2} );
  //                                z  y  h  g  f  x  e  i
  at::Tensor l_out_ref = at::randn( {2, 4, 4, 6, 5, 3, 8, 3} );
  at::Tensor l_out = l_out_ref.clone();

  at::Tensor l_left_aos    = at::view_as_complex( l_left.permute(  { 1, 2, 3, 4, 5, 6, 7, 0 } ).contiguous() );
  at::Tensor l_right_aos   = at::view_as_complex( l_right.permute( { 1, 2, 3, 4, 5, 6, 0 } ).contiguous() );
  at::Tensor l_out_ref_aos = at::view_as_complex( l_out_ref.permute(   { 1, 2, 3, 4, 5, 6, 7, 0 } ).contiguous() );

  // reference
  l_out_ref = at::einsum( "yxgcaei,yxhfca->yhgfxei",
                          {l_left_aos, l_right_aos} );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( { 1, 2, 3, 4, 5, 6, 7, 0 } ).contiguous() );
  REQUIRE( at::allclose( l_out_aos, l_out_ref, 1E-4, 1E-6 )  );
}

TEST_CASE( "FP32 BLAS-based binary contraction involving C, M, N and K dimensions, stride-1 C.", "[binary_contraction_blas]" ) {
  // Test case:
  //
  //         ______________hgfxeiy________________
  //        /                                     \
  //   xgycaei                                   xhyfca
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
  //  hgfxeiy: 6 4 5 7 1 0 8
  //  xgycaei: 7 4 8 3 2 1 0
  //  xhyfca:  7 6 8 5 3 2
  //
  //   dim types:
  //     c:  yx /  87
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32
  //
  // BLAS call will use blocking:
  //   cb: y
  //   mb: e, i
  //   nb: f
  //   kb: c, a
  // ordering:
  //   left  (BC-BM-BK-CB-KB-MB): x - g - - y - ca - ei
  //   right (BC-BN-BK-CB-NB-KB): x - h - - y - f  - ca

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

  int64_t l_dim_ids_out[7] = { 6, 4, 5, 7, 1, 0, 8 };
  int64_t l_dim_ids_left[7] = { 7, 4, 8, 3, 2, 1, 0 };
  int64_t l_dim_ids_right[6] = { 7, 6, 8, 5, 3, 2 };

  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 7,
                   6,
                   7,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  //                              x  g  y  c  a  e  i
  at::Tensor l_left = at::randn( {3, 6, 4, 7, 2, 8, 3} );
  //                               x  h  y  f  c  a
  at::Tensor l_right = at::randn( {3, 4, 4, 5, 7, 2} );
  //                                h  g  f  x  e  i  y
  at::Tensor l_out_ref = at::randn( {4, 6, 5, 3, 8, 3, 4} );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "xgycaei,xhyfca->hgfxeiy",
                          {l_left, l_right} );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_native, l_out_ref, 1E-4, 1E-6 )  );
}

TEST_CASE( "Complex FP32 BLAS-based binary contraction involving C, M, N and K dimensions, stride-1 C.", "[binary_contraction_blas]" ) {
  // Test case:
  //
  //         _____________zhgfxeiy________________
  //        /                                     \
  //  zxgycaei                                  zxhyfca
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
  //      z  9    2   c2
  //
  //  zyhgfxei: 9 8 6 4 5 7 1 0
  //  zyxgcaei: 9 8 7 4 3 2 1 0
  //  zyxhfca:  9 8 7 6 5 3 2
  //
  //   dim types:
  //     c: zyx / 287
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32
  //
  // BLAS call will use blocking:
  //   cb: y
  //   mb: e, i
  //   nb: f
  //   kb: c, a
  // ordering:
  //   left  (BC-BM-BK-CB-KB-MB): x - g - - y - ca - ei
  //   right (BC-BN-BK-CB-NB-KB): x - h - - y - f  - ca

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
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 9, 2 ) );

  int64_t l_dim_ids_out[8] = { 9, 6, 4, 5, 7, 1, 0, 8 };
  int64_t l_dim_ids_left[8] = { 9, 7, 4, 8, 3, 2, 1, 0 };
  int64_t l_dim_ids_right[7] = { 9, 7, 6, 8, 5, 3, 2 };

  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 8,
                   7,
                   8,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::CPX_ZERO,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  //                              z  x  g  y  c  a  e  i
  at::Tensor l_left = at::randn( {2, 3, 6, 4, 7, 2, 8, 3} );
  //                               z  x  h  y  f  c  a
  at::Tensor l_right = at::randn( {2, 3, 4, 4, 5, 7, 2} );
  //                                z  h  g  f  x  e  i  y
  at::Tensor l_out_ref = at::randn( {2, 4, 6, 5, 3, 8, 3, 4} );
  at::Tensor l_out = l_out_ref.clone();

  at::Tensor l_left_aos    = at::view_as_complex( l_left.permute(  { 1, 2, 3, 4, 5, 6, 7, 0 } ).contiguous() );
  at::Tensor l_right_aos   = at::view_as_complex( l_right.permute( { 1, 2, 3, 4, 5, 6, 0 } ).contiguous() );
  at::Tensor l_out_ref_aos = at::view_as_complex( l_out_ref.permute( { 1, 2, 3, 4, 5, 6, 7, 0 } ).contiguous() );

  // reference
  l_out_ref = at::einsum( "xgycaei,xhyfca->hgfxeiy",
                          {l_left_aos, l_right_aos} );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out.data_ptr() );

  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( { 1, 2, 3, 4, 5, 6, 7, 0 } ).contiguous() );
  REQUIRE( at::allclose( l_out_aos, l_out_ref, 1E-4, 1E-6 )  );
}