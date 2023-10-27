#include "catch.hpp"
#include "Unary.h"

TEST_CASE( "Stride derivation.", "[unary]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in[3]  = {0, 1, 2};
  int64_t l_dim_ids_out[3] = {0, 2, 1};

  int64_t l_strides_in[3]  = {0};
  int64_t l_strides_out[3] = {0};

  einsum_ir::backend::Unary::strides( 3,
                                      &l_dim_sizes,
                                      l_dim_ids_in,
                                      l_dim_ids_out,
                                      l_strides_in,
                                      l_strides_out );

  REQUIRE( l_strides_in[0]  == 12 );
  REQUIRE( l_strides_in[1]  ==  1 );
  REQUIRE( l_strides_in[2]  ==  4 );

  REQUIRE( l_strides_out[0] == 12 );
  REQUIRE( l_strides_out[1] ==  3 );
  REQUIRE( l_strides_out[2] ==  1 );
}