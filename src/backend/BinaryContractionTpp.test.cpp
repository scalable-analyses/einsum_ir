#include "catch.hpp"
#include "BinaryContractionTpp.h"

TEST_CASE( "Tests the left/right swap of TPP-based binary contractions.", "[bin_cont_tpp_swap]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

  // native data layout
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
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );
  l_bin_cont.compile();

  REQUIRE( l_bin_cont.m_tensors_in_swapped == false );
  REQUIRE( l_bin_cont.m_strides_in_left_m[0] == 1 );
  REQUIRE( l_bin_cont.m_strides_in_left_k[0] == 2 );
  REQUIRE( l_bin_cont.m_strides_in_right_n[0] == 4 );
  REQUIRE( l_bin_cont.m_strides_in_right_k[0] == 1 );
  REQUIRE( l_bin_cont.m_strides_out_m[0] == 1 );
  REQUIRE( l_bin_cont.m_strides_out_n[0] == 2 );

  // swap required
  //
  //    ____mn___
  //   /         \
  // mk           kn
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  int64_t l_dim_ids_in_left_swap[2]  = { 0, 2 };
  int64_t l_dim_ids_in_right_swap[2] = { 2, 1 };
  int64_t l_dim_ids_out_swap[2] = { 0, 1 };
  l_bin_cont.init( 2,
                   2,
                   2,
                   l_dim_sizes,
                   l_dim_ids_in_left_swap,
                   l_dim_ids_in_right_swap,
                   l_dim_ids_out_swap,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );
  l_bin_cont.compile();
  REQUIRE( l_bin_cont.m_tensors_in_swapped == true );
  REQUIRE( l_bin_cont.m_strides_in_left_m[0] == 1 );
  REQUIRE( l_bin_cont.m_strides_in_left_k[0] == 3 );
  REQUIRE( l_bin_cont.m_strides_in_right_n[0] == 4 );
  REQUIRE( l_bin_cont.m_strides_in_right_k[0] == 1 );
  REQUIRE( l_bin_cont.m_strides_out_m[0] == 1 );
  REQUIRE( l_bin_cont.m_strides_out_n[0] == 3 );
}