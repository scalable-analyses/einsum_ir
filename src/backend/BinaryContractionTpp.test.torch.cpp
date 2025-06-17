#include "ATen/ATen.h"
#include "catch.hpp"
#include "BinaryContractionTpp.h"

#ifdef _OPENMP
#include <omp.h>
#endif

TEST_CASE( "TPP-based binary contraction executing matmuls.", "[binary_contraction_tpp]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

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
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );

  // data
  at::Tensor l_in_left  = at::randn( {4, 2} );
  at::Tensor l_in_right = at::randn( {3, 4} );
  at::Tensor l_out_ref  = at::randn( {3, 2} );
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

TEST_CASE( "TPP-based Matrix-matrix multiplication with a full-tensor bias.", "[binary_contraction_tpp]" ) {
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

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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
                   einsum_ir::COPY,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );
  // data
  at::Tensor l_in_left  = at::randn( {4, 2} );
  at::Tensor l_in_right = at::randn( {3, 4} );
  at::Tensor l_bias     = at::randn( {3, 2} );
  at::Tensor l_out_ref  = at::randn( {3, 2} );
  at::Tensor l_out      = at::randn( {3, 2} );

  // reference
  l_out_ref = l_bias + at::einsum( "km,nk->nm",
                                   {l_in_left, l_in_right} );

  // native input dimensions
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-7 )  );
}

TEST_CASE( "TPP-based matrix-matrix multiplication with a bias (scalar to matrix bcast).", "[binary_contraction_tpp]" ) {
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

  std::map< int64_t, int64_t > l_dim_sizes_out_aux;
  l_dim_sizes_out_aux.insert( std::pair< int64_t, int64_t >( 0, 1 ) );
  l_dim_sizes_out_aux.insert( std::pair< int64_t, int64_t >( 1, 1 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   2,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes_out_aux,
                   &l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::COPY,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );
  // data
  at::Tensor l_in_left  = at::randn( {4, 2} );
  at::Tensor l_in_right = at::randn( {3, 4} );
  at::Tensor l_bias     = at::randn( {1, 1} );
  at::Tensor l_out_ref  = at::randn( {3, 2} );
  at::Tensor l_out      = at::randn( {3, 2} );

  // reference
  l_out_ref = l_bias + at::einsum( "km,nk->nm",
                                   {l_in_left, l_in_right} );

  // native input dimensions
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "FP32 TPP-based matrix-matrix multiplication with a bias (row to matrix bcast).", "[binary_contraction_tpp]" ) {
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

  std::map< int64_t, int64_t > l_dim_sizes_out_aux;
  l_dim_sizes_out_aux.insert( std::pair< int64_t, int64_t >( 0, 1 ) );
  l_dim_sizes_out_aux.insert( std::pair< int64_t, int64_t >( 1, 3 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   2,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes_out_aux,
                   &l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::COPY,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );
  // data
  at::Tensor l_in_left  = at::randn( {4, 2} );
  at::Tensor l_in_right = at::randn( {3, 4} );
  at::Tensor l_bias     = at::randn( {3, 1} );
  at::Tensor l_out_ref  = at::randn( {3, 2} );
  at::Tensor l_out      = at::randn( {3, 2} );

  // reference
  l_out_ref = l_bias + at::einsum( "km,nk->nm",
                                   {l_in_left, l_in_right} );

  // native input dimensions
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out,
                         l_out_ref,
                         1.0e-4,
                         1.0e-7 )  );
}

TEST_CASE( "TPP-based matrix-matrix multiplication with a bias (column to matrix bcast).", "[binary_contraction_tpp]" ) {
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

  std::map< int64_t, int64_t > l_dim_sizes_out_aux;
  l_dim_sizes_out_aux.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes_out_aux.insert( std::pair< int64_t, int64_t >( 1, 1 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   2,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes_out_aux,
                   &l_dim_sizes,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::COPY,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );
  // data
  at::Tensor l_in_left  = at::randn( {4, 2} );
  at::Tensor l_in_right = at::randn( {3, 4} );
  at::Tensor l_bias     = at::randn( {1, 2} );
  at::Tensor l_out_ref  = at::randn( {3, 2} );
  at::Tensor l_out      = at::randn( {3, 2} );

  // reference
  l_out_ref = l_bias + at::einsum( "km,nk->nm",
                                   {l_in_left, l_in_right} );

  // native input dimensions
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "FP32 TPP-based binary contraction executing a batched matmul.", "[binary_contraction_tpp]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 5 ) );

  int64_t l_dim_ids_in_left[3]  = { 3, 1, 0 };
  int64_t l_dim_ids_in_right[3] = { 2, 3, 0 };
  int64_t l_dim_ids_out[3]      = { 2, 1, 0 };

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

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
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );

  // data
  at::Tensor l_in_left  = at::randn( {5, 3, 2} );
  at::Tensor l_in_right = at::randn( {4, 5, 2} );
  at::Tensor l_out_ref  = at::randn( {4, 3, 2} );
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

  REQUIRE( at::allclose( l_out_ref,
                         l_out_native,
                         1.0e-4,
                         1.0e-7 )  );
}

TEST_CASE( "TPP-based binary contraction executing matmuls with FP64 and zero first touch.", "[binary_contraction_tpp]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

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
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );

  // data
  at::Tensor l_in_left  = at::randn( {4, 2},
                                    at::ScalarType::Double );
  at::Tensor l_in_right = at::randn( {3, 4},
                                    at::ScalarType::Double );
  at::Tensor l_out_ref  = at::randn( {3, 2},
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

TEST_CASE( "FP32 TPP-based binary contraction involving C, M, N and K dimensions, stride-1 M.", "[binary_contraction_tpp]" ) {
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

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );

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

  REQUIRE( at::allclose( l_out_native, l_out_ref, 1E-4, 1E-5 )  );
}


TEST_CASE( "FP32 TPP-based binary contraction involving C, M, N and K dimensions, stride-1 M with packing.", "[binary_contraction_tpp_packing]" ) {
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
  //  pack_left:  8 7  3 2  4 1 0
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

  int64_t l_dim_ids_pack_left[7] = { 8, 7, 3, 2, 4, 1, 0 };

  einsum_ir::backend::MemoryManager l_memory;

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init( 7,
                   6,
                   7,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   nullptr,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_dim_ids_out,
                   l_dim_ids_pack_left,
                   nullptr,
                   &l_memory,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );

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
  l_memory.alloc_all_memory();

  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_native, l_out_ref, 1E-4, 1E-5 )  );
}

TEST_CASE( "FP32 TPP-based binary contraction involving C, M, N and K dimensions, stride-1 M, zero first-touch op, ReLU last-touch op.", "[binary_contraction_tpp]" ) {
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

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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
                   einsum_ir::RELU,
                   l_num_threads );

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
  l_out_ref = at::relu( l_out_ref );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_native, l_out_ref, 1E-4, 1E-6 )  );
}

TEST_CASE( "FP32 TPP-based binary contraction involving C, M, N and K dimensions, stride-1 C.", "[binary_contraction_tpp]" ) {
  // Test case:
  //
  //         ______________hgfxeiy________________
  //        /                                     \
  //   xgcaeiy                                   xhfcay
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
  //  xgcaeiy: 7 4 3 2 1 0 8
  //  xhfcay:  7 6 5 3 2 8
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
  int64_t l_dim_ids_left[7] = { 7, 4, 3, 2, 1, 0, 8 };
  int64_t l_dim_ids_right[6] = { 7, 6, 5, 3, 2, 8 };

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
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
                   einsum_ir::UNDEFINED_KTYPE,
                   l_num_threads );

  //                              x  g  c  a  e  i  y
  at::Tensor l_left = at::randn( {3, 6, 7, 2, 8, 3, 4} );
  //                               x  h  f  c  a  y
  at::Tensor l_right = at::randn( {3, 4, 5, 7, 2, 4} );
  //                                h  g  f  x  e  i  y
  at::Tensor l_out_ref = at::randn( {4, 6, 5, 3, 8, 3, 4} );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "xgcaeiy,xhfcay->hgfxeiy",
                          {l_left, l_right} );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_bin_cont.contract( l_left.data_ptr(),
                       l_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_native, l_out_ref, 1E-3, 1E-5 )  );
}