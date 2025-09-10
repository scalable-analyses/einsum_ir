#include "ATen/ATen.h"
#include "catch.hpp"
#include "UnaryScalar.h"

TEST_CASE( "Small tensor transposition through the scalar unary interface using FP64 data.", "[unary_scalar]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_t0[3] = { 0, 1, 2 };
  int64_t l_dim_ids_t1[3] = { 2, 1, 0 };

  einsum_ir::backend::UnaryScalar l_unary_scalar;

  l_unary_scalar.init( 3,
                       &l_dim_sizes,
                       l_dim_ids_t0,
                       l_dim_ids_t1,
                       einsum_ir::data_t::FP64,
                       einsum_ir::data_t::FP64,
                       einsum_ir::data_t::FP64,
                       einsum_ir::kernel_t::COPY,
                       1 );

  l_unary_scalar.compile();

  at::Tensor l_t0 = at::randn( {3, 5, 4},
                               at::ScalarType::Double );

  at::Tensor l_t1 = at::randn( {4, 5, 3},
                               at::ScalarType::Double );

  l_unary_scalar.eval( l_t0.data_ptr(),
                       l_t1.data_ptr() );

  REQUIRE( at::equal( l_t0.permute( {2, 1, 0} ), l_t1 ) );
}

TEST_CASE( "Large tensor transposition through the scalar unary interface using FP32 data.", "[unary_scalar]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 7 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 5, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 6, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 7, 8 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 8, 6 ) );

  int64_t l_dim_ids_t0[9] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
  int64_t l_dim_ids_t1[9] = { 2, 1, 4, 0, 5, 7, 3, 8, 6 };

  einsum_ir::backend::UnaryScalar l_unary_scalar;

  l_unary_scalar.init( 9,
                       &l_dim_sizes,
                       l_dim_ids_t0,
                       l_dim_ids_t1,
                       einsum_ir::data_t::FP32,
                       einsum_ir::data_t::FP32,
                       einsum_ir::data_t::FP32,
                       einsum_ir::kernel_t::COPY,
                       9 );

  l_unary_scalar.compile();

  //                            0  1  2  3  4  5  6  7  8
  at::Tensor l_t0 = at::randn( {3, 5, 4, 7, 2, 5, 3, 8, 6},
                               at::ScalarType::Float );

  at::Tensor l_t1 = at::randn( {4, 5, 2, 3, 5, 8, 7, 6, 3},
                               at::ScalarType::Float );

  l_unary_scalar.eval( l_t0.data_ptr(),
                       l_t1.data_ptr() );

  REQUIRE( at::equal( l_t0.permute( {2, 1, 4, 0, 5, 7, 3, 8, 6} ), l_t1 ) );
}