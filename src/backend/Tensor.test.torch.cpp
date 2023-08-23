#include <ATen/ATen.h>
#include "catch.hpp"
#include "Tensor.h"

TEST_CASE( "Tests the permutation function using a permutation array.", "[tensor_permute]" ) {
  //                            0  1  2  3
  at::Tensor l_ten = at::rand( {2, 4, 3, 5} );

  at::Tensor l_ten_perm = at::rand( {3, 2, 5, 4} );

  at::Tensor l_ten_perm_ref = l_ten.permute( { 2, 0, 3, 1  } ).contiguous();

  int64_t l_sizes[4] = { 2, 4, 3, 5 };
  int64_t l_perm[4] = { 2, 0, 3, 1 };

  einsum_ir::backend::Tensor::permute( 4,
                                       l_sizes,
                                       l_perm,
                                       einsum_ir::FP32,
                                       einsum_ir::FP32,
                                       l_ten.data_ptr(),
                                       l_ten_perm.data_ptr() );

  REQUIRE( at::equal( l_ten_perm_ref, l_ten_perm ) );
}

TEST_CASE( "Tests the permutation function using dimension ids.", "[tensor_permute]" ) {
  //                            0  1  2  3
  at::Tensor l_ten = at::rand( {2, 4, 3, 5} );

  at::Tensor l_ten_perm = at::rand( {3, 2, 5, 4} );

  at::Tensor l_ten_perm_ref = l_ten.permute( { 2, 0, 3, 1  } ).contiguous();

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 10, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 11, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 12, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 13, 5 ) );

  int64_t l_dim_ids_in[4] =  { 10, 11, 12, 13 };
  int64_t l_dim_ids_out[4] = { 12, 10, 13, 11 };

  einsum_ir::backend::Tensor::permute( 4,
                                       l_dim_sizes,
                                       l_dim_ids_in,
                                       l_dim_ids_out,
                                       einsum_ir::FP32,
                                       einsum_ir::FP32,
                                       l_ten.data_ptr(),
                                       l_ten_perm.data_ptr() );

  REQUIRE( at::equal( l_ten_perm_ref, l_ten_perm ) );
}