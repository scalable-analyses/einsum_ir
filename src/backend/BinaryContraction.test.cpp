#include "catch.hpp"
#include "BinaryContraction.h"

TEST_CASE( "Derives the dimension types given a binary einsum string of int64 values", "[bin_cont_dim_types]" ) {
  //         ______________iexfghy________________
  //        /                                     \
  //   ie-a-x-c-g-y                            a-f-x-c-h-y
  //
  // i: 0
  // e: 1
  // x: 2
  // f: 3
  // g: 4
  // h: 5
  // y: 6
  // a: 7
  // c: 8
  int64_t l_ids_out[7] = { 0, 1, 2, 3, 4, 5, 6 };
  int64_t l_ids_in_left[7] = { 0, 1, 7, 2, 8, 4, 6 };
  int64_t l_ids_in_right[6] = { 7, 3, 2, 8, 5, 6 };

  einsum_ir::dim_t l_types_out[7] = { einsum_ir::UNDEFINED_DIM };
  einsum_ir::dim_t l_types_in_left[7] = { einsum_ir::UNDEFINED_DIM };
  einsum_ir::dim_t l_types_in_right[6] = { einsum_ir::UNDEFINED_DIM };

  einsum_ir::backend::BinaryContraction::dim_types( 7,
                                                    6,
                                                    7,
                                                    l_ids_in_left,
                                                    l_ids_in_right,
                                                    l_ids_out,
                                                    einsum_ir::UNDEFINED_DIM,
                                                    einsum_ir::M,
                                                    einsum_ir::N,
                                                    einsum_ir::C,
                                                    l_types_out );

  REQUIRE( l_types_out[0] == einsum_ir::M );
  REQUIRE( l_types_out[1] == einsum_ir::M );
  REQUIRE( l_types_out[2] == einsum_ir::C );
  REQUIRE( l_types_out[3] == einsum_ir::N );
  REQUIRE( l_types_out[4] == einsum_ir::M );
  REQUIRE( l_types_out[5] == einsum_ir::N );
  REQUIRE( l_types_out[6] == einsum_ir::C );

  einsum_ir::backend::BinaryContraction::dim_types( 6,
                                                    7,
                                                    7,
                                                    l_ids_in_right,
                                                    l_ids_out,
                                                    l_ids_in_left,
                                                    einsum_ir::I,
                                                    einsum_ir::K,
                                                    einsum_ir::M,
                                                    einsum_ir::C,
                                                    l_types_in_left );

  REQUIRE( l_types_in_left[0] == einsum_ir::M );
  REQUIRE( l_types_in_left[1] == einsum_ir::M );
  REQUIRE( l_types_in_left[2] == einsum_ir::K );
  REQUIRE( l_types_in_left[3] == einsum_ir::C );
  REQUIRE( l_types_in_left[4] == einsum_ir::K );
  REQUIRE( l_types_in_left[5] == einsum_ir::M );
  REQUIRE( l_types_in_left[6] == einsum_ir::C );

  einsum_ir::backend::BinaryContraction::dim_types( 7,
                                                    7,
                                                    6,
                                                    l_ids_in_left,
                                                    l_ids_out,
                                                    l_ids_in_right,
                                                    einsum_ir::J,
                                                    einsum_ir::K,
                                                    einsum_ir::N,
                                                    einsum_ir::C,
                                                    l_types_in_right );

  REQUIRE( l_types_in_right[0] == einsum_ir::K );
  REQUIRE( l_types_in_right[1] == einsum_ir::N );
  REQUIRE( l_types_in_right[2] == einsum_ir::C );
  REQUIRE( l_types_in_right[3] == einsum_ir::K );
  REQUIRE( l_types_in_right[4] == einsum_ir::N );
  REQUIRE( l_types_in_right[5] == einsum_ir::C );
}

TEST_CASE( "Filters the dimension ids based on the given type.", "[filter_dim_ids]" ) {
  int64_t l_ids_out[7] = { 10, 11, 12, 13, 14, 15, 16 };
  einsum_ir::dim_t l_types_out[7] = { einsum_ir::M,   // 10
                                      einsum_ir::M,   // 11
                                      einsum_ir::C,   // 12
                                      einsum_ir::N,   // 13
                                      einsum_ir::M,   // 14
                                      einsum_ir::N,   // 15
                                      einsum_ir::C }; // 16

  int64_t l_ids_m[3] = { 0 };
  int64_t l_ids_n[2] = { 0 };
  int64_t l_ids_c[2] = { 0 };

  int64_t l_num_dims_m = einsum_ir::backend::BinaryContraction::filter_dim_ids( 7,
                                                                                einsum_ir::M,
                                                                                l_types_out,
                                                                                l_ids_out,
                                                                                l_ids_m );
  REQUIRE( l_num_dims_m == 3 );
  REQUIRE( l_ids_m[0] == 10 );
  REQUIRE( l_ids_m[1] == 11 );
  REQUIRE( l_ids_m[2] == 14 );

  int64_t l_num_dims_n = einsum_ir::backend::BinaryContraction::filter_dim_ids( 7,
                                                                                einsum_ir::N,
                                                                                l_types_out,
                                                                                l_ids_out,
                                                                                l_ids_n );
  REQUIRE( l_num_dims_n == 2 );
  REQUIRE( l_ids_n[0] == 13 );
  REQUIRE( l_ids_n[1] == 15 );

  int64_t l_num_dims_c = einsum_ir::backend::BinaryContraction::filter_dim_ids( 7,
                                                                                einsum_ir::C,
                                                                                l_types_out,
                                                                                l_ids_out,
                                                                                l_ids_c );
  REQUIRE( l_num_dims_c == 2 );
  REQUIRE( l_ids_c[0] == 12 );
  REQUIRE( l_ids_c[1] == 16 );
}