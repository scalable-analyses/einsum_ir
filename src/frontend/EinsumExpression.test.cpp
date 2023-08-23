#include "catch.hpp"
#include "EinsumExpression.h"

TEST_CASE( "Derivation of dimension histogram.", "[einsum_exp]" ) {
  int64_t l_string_dim_ids[8] = { 0, 2, 3, 1, 0, 4, 0, 2 };
  int64_t l_histogram[5];

  for( int64_t l_di = 0; l_di < 5; l_di++ ) {
    l_histogram[l_di] = -1;
  }

  einsum_ir::frontend::EinsumExpression::histogram( 5,
                                                    8,
                                                    l_string_dim_ids,
                                                    l_histogram );

  REQUIRE( l_histogram[0] == 3 );
  REQUIRE( l_histogram[1] == 1 );
  REQUIRE( l_histogram[2] == 2 );
  REQUIRE( l_histogram[3] == 1 );
  REQUIRE( l_histogram[4] == 1 );
}

TEST_CASE( "Derivation of output substrings.", "[einsum_exp]" ) {
  int64_t l_dim_ids_left[4]  = { 1, 4, 2, 0 };
  int64_t l_dim_ids_right[3] = { 0, 5, 2 };
  //                         0  1  2  3  4  5
  int64_t l_histogram[6] = { 3, 5, 2, 7, 9, 2 };

  std::vector< int64_t > l_substring_out;

  einsum_ir::frontend::EinsumExpression::substring_out( 4,
                                                        3,
                                                        l_dim_ids_left,
                                                        l_dim_ids_right,
                                                        l_histogram,
                                                        l_substring_out );

  REQUIRE( l_substring_out.size() == 4 );
  REQUIRE( l_substring_out[0] == 0 );
  REQUIRE( l_substring_out[1] == 1 );
  REQUIRE( l_substring_out[2] == 4 );
  REQUIRE( l_substring_out[3] == 5 );
}

TEST_CASE( "Unique contraction path generation.", "[einsum_exp]" ) {
  int64_t l_path[6] = { 1, 2,  2, 0,  0, 1 };
  int64_t l_path_unique[6] = { 0 };

  einsum_ir::frontend::EinsumExpression::unique_tensor_ids( 3,
                                                            l_path,
                                                            l_path_unique );

  REQUIRE( l_path_unique[0] == 1 );
  REQUIRE( l_path_unique[1] == 2 );

  REQUIRE( l_path_unique[2] == 4 );
  REQUIRE( l_path_unique[3] == 0 );

  REQUIRE( l_path_unique[4] == 3 );
  REQUIRE( l_path_unique[5] == 5 );
}