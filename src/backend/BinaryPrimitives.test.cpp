#include "catch.hpp"
#include "BinaryPrimitives.h"

TEST_CASE( "Swap inputs for ab,ca->cb", "[binary_primitives]" ) {
  int64_t l_dim_ids_left[ 2 ]  = { 'a', 'b' };
  int64_t l_dim_ids_right[ 2 ] = { 'c', 'a' };
  int64_t l_dim_ids_out[ 2 ]   = { 'c', 'b' };

  bool l_swap = einsum_ir::backend::BinaryPrimitives::swap_inputs( 2,
                                                                   2,
                                                                   2,
                                                                   l_dim_ids_left,
                                                                   l_dim_ids_right,
                                                                   l_dim_ids_out );

  REQUIRE( l_swap == false );
}

TEST_CASE( "Swap inputs for ca,ab->cb", "[binary_primitives]" ) {
  int64_t l_dim_ids_left[ 2 ]  = { 'c', 'a' };
  int64_t l_dim_ids_right[ 2 ] = { 'a', 'b' };
  int64_t l_dim_ids_out[ 2 ]   = { 'c', 'b' };

  bool l_swap = einsum_ir::backend::BinaryPrimitives::swap_inputs( 2,
                                                                   2,
                                                                   2,
                                                                   l_dim_ids_left,
                                                                   l_dim_ids_right,
                                                                   l_dim_ids_out );

  REQUIRE( l_swap == true );
}

TEST_CASE( "Swap inputs for abd,cad->cbd", "[binary_primitives]" ) {
  int64_t l_dim_ids_left[ 3 ]  = { 'a', 'b', 'd' };
  int64_t l_dim_ids_right[ 3 ] = { 'c', 'a', 'd' };
  int64_t l_dim_ids_out[ 3 ]   = { 'c', 'b', 'd' };

  bool l_swap = einsum_ir::backend::BinaryPrimitives::swap_inputs( 3,
                                                                   3,
                                                                   3,
                                                                   l_dim_ids_left,
                                                                   l_dim_ids_right,
                                                                   l_dim_ids_out );

  REQUIRE( l_swap == false );
}

TEST_CASE( "Swap inputs for cad,abd->cbd", "[binary_primitives]" ) {
  int64_t l_dim_ids_left[ 3 ]  = { 'c', 'a', 'd' };
  int64_t l_dim_ids_right[ 3 ] = { 'a', 'b', 'd' };
  int64_t l_dim_ids_out[ 3 ]   = { 'c', 'b', 'd' };

  bool l_swap = einsum_ir::backend::BinaryPrimitives::swap_inputs( 3,
                                                                   3,
                                                                   3,
                                                                   l_dim_ids_left,
                                                                   l_dim_ids_right,
                                                                   l_dim_ids_out );

  REQUIRE( l_swap == true );
}

TEST_CASE( "Blocking of a matrix-matrix multiplication. left: kb x mb, right: nb x kb, out: nb x mb", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );

  int64_t l_dim_ids_left[ 2 ]  = { 'a', 'b' };
  int64_t l_dim_ids_right[ 2 ] = { 'c', 'a' };
  int64_t l_dim_ids_out[ 2 ]   = { 'c', 'b' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;

  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          2,
                          2,
                          2,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 0 );
  REQUIRE( l_dim_ids_mb.size() == 1 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );

  REQUIRE( l_dim_ids_mb[ 0 ] == 'b' );
  REQUIRE( l_dim_ids_nb[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_kb[ 0 ] == 'a' );
}

TEST_CASE( "Blocking of a matrix-matrix multiplication with non-contiguous memory layout.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 2 ) );

  int64_t l_dim_ids_left[ 4 ]  = { 'a', 'b', 'c', 'd' };
  int64_t l_dim_ids_right[ 4 ] = { 'e', 'f', 'a', 'b' };
  int64_t l_dim_ids_out[ 4 ]   = { 'e', 'f', 'c', 'd' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  std::map< int64_t, int64_t > l_strides_left;
  l_strides_left.insert( std::pair< int64_t, int64_t >( 'd', 1 ) );
  l_strides_left.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );
  l_strides_left.insert( std::pair< int64_t, int64_t >( 'b', 4 ) );
  l_strides_left.insert( std::pair< int64_t, int64_t >( 'a', 8 ) );

  std::map< int64_t, int64_t > l_strides_right;
  l_strides_right.insert( std::pair< int64_t, int64_t >( 'b', 3 ) );
  l_strides_right.insert( std::pair< int64_t, int64_t >( 'a', 6 ) );
  l_strides_right.insert( std::pair< int64_t, int64_t >( 'f', 12 ) );
  l_strides_right.insert( std::pair< int64_t, int64_t >( 'e', 24 ) );

  std::map< int64_t, int64_t > l_strides_out;
  l_strides_out.insert( std::pair< int64_t, int64_t >( 'd', 1 ) );
  l_strides_out.insert( std::pair< int64_t, int64_t >( 'c', 3 ) );
  l_strides_out.insert( std::pair< int64_t, int64_t >( 'f', 6 ) );
  l_strides_out.insert( std::pair< int64_t, int64_t >( 'e', 12 ) );

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;

  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          4,
                          4,
                          4,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          &l_strides_left,
                          &l_strides_right,
                          &l_strides_out,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 0 );
  REQUIRE( l_dim_ids_mb.size() == 1 );
  REQUIRE( l_dim_ids_nb.size() == 2 );
  REQUIRE( l_dim_ids_kb.size() == 0 );

  REQUIRE( l_dim_ids_mb[ 0 ] == 'd' );

  REQUIRE( l_dim_ids_nb[ 0 ] == 'e' );
  REQUIRE( l_dim_ids_nb[ 1 ] == 'f' );
}

TEST_CASE( "Blocking of the binary contraction: 12 13 2 3 9 10 11 14 , 0 9 10 1 11 -> 0 1 12 13 14", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,
              16,
              32,
              128,
              12,
              64,
              32,
              512 );

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  0, 32 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  1, 96 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  2,  3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  3,  3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  4,  3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  5,  3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  6,  2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  7,  2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  8, 64 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >(  9,  2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 10,  2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 11, 64 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 12,  2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 13,  2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 14, 64 ) );

  int64_t l_dim_ids_left[ 8 ]  = { 12, 13, 2, 3, 9, 10, 11, 14 };
  int64_t l_dim_ids_right[ 5 ] = { 0, 9, 10, 1, 11 };
  int64_t l_dim_ids_out[ 5 ]   = { 0, 1, 12, 13, 14 };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          8,
                          5,
                          5,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 0 );
  REQUIRE( l_dim_ids_mb.size() == 1 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );
}

TEST_CASE( "Blocking of TCCG setting #1: efbad,cf->abcde 48,36,24,36,48,36.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );

  int64_t l_dim_ids_left[ 5 ]  = { 'e', 'f', 'b', 'a', 'd' };
  int64_t l_dim_ids_right[ 2 ] = { 'c', 'f' };
  int64_t l_dim_ids_out[ 5 ]   = { 'a', 'b', 'c', 'd', 'e' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          5,
                          2,
                          5,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 0 );
  REQUIRE( l_dim_ids_mb.size() == 0 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );

  REQUIRE( l_dim_ids_nb[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_kb[ 0 ] == 'f' );
}

TEST_CASE( "Blocking of TCCG setting #1 (reordered, small M): fbade,cf->abcde 48,36,24,36,48,36.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );

  int64_t l_dim_ids_left[ 5 ]  = { 'f', 'b', 'a', 'd', 'e' };
  int64_t l_dim_ids_right[ 2 ] = { 'c', 'f' };
  int64_t l_dim_ids_out[ 5 ]   = { 'a', 'b', 'c', 'd', 'e' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          5,
                          2,
                          5,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 0 );
  REQUIRE( l_dim_ids_mb.size() == 1 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );

  REQUIRE( l_dim_ids_mb[ 0 ] == 'e' );
  REQUIRE( l_dim_ids_nb[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_kb[ 0 ] == 'f' );
}

TEST_CASE( "Blocking of TCCG setting #1 with batch: efbadx,cfx->abcdex 48,36,24,36,48,36,16", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'x', 16 ) );

  int64_t l_dim_ids_left[ 6 ]  = { 'e', 'f', 'b', 'a', 'd', 'x' };
  int64_t l_dim_ids_right[ 3 ] = { 'c', 'f', 'x' };
  int64_t l_dim_ids_out[ 6 ]   = { 'a', 'b', 'c', 'd', 'e', 'x' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          6,
                          3,
                          6,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 1 );
  REQUIRE( l_dim_ids_mb.size() == 0 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );

  REQUIRE( l_dim_ids_cb[ 0 ] == 'x' );
  REQUIRE( l_dim_ids_nb[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_kb[ 0 ] == 'f' );
}

TEST_CASE( "Blocking of TCCG setting #1 with batch (reordered, small M): fbadex,cfx->abcdex 48,36,24,36,48,36,16", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'x', 16 ) );

  int64_t l_dim_ids_left[ 6 ]  = { 'f', 'b', 'a', 'd', 'e', 'x' };
  int64_t l_dim_ids_right[ 3 ] = { 'c', 'f', 'x' };
  int64_t l_dim_ids_out[ 6 ]   = { 'a', 'b', 'c', 'd', 'e', 'x' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          6,
                          3,
                          6,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 1 );
  REQUIRE( l_dim_ids_mb.size() == 1 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );

  REQUIRE( l_dim_ids_cb[ 0 ] == 'x' );
  REQUIRE( l_dim_ids_mb[ 0 ] == 'e' );
  REQUIRE( l_dim_ids_nb[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_kb[ 0 ] == 'f' );
}

TEST_CASE( "Blocking of TCCG setting #1 with batch (reordered, large M): fbadex,cfx->abcdex 48,36,24,36,48,36,16", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,     8,   // C
              32, 4096,   // M
              16,   64,   // N
              64,  512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'x', 16 ) );

  int64_t l_dim_ids_left[ 6 ]  = { 'f', 'b', 'a', 'd', 'e', 'x' };
  int64_t l_dim_ids_right[ 3 ] = { 'c', 'f', 'x' };
  int64_t l_dim_ids_out[ 6 ]   = { 'a', 'b', 'c', 'd', 'e', 'x' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          6,
                          3,
                          6,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 1 );
  REQUIRE( l_dim_ids_mb.size() == 2 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );

  REQUIRE( l_dim_ids_cb[ 0 ] == 'x' );
  REQUIRE( l_dim_ids_mb[ 0 ] == 'd' );
  REQUIRE( l_dim_ids_mb[ 1 ] == 'e' );
  REQUIRE( l_dim_ids_nb[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_kb[ 0 ] == 'f' );
}

TEST_CASE( "Blocking of TCCG setting #1 (reordered, large M): fbade,cf->abcde 48,36,24,36,48,36.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,     8,   // C
              32, 4096,   // M
              16,   64,   // N
              64,  512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );

  int64_t l_dim_ids_left[ 5 ]  = { 'f', 'b', 'a', 'd', 'e' };
  int64_t l_dim_ids_right[ 2 ] = { 'c', 'f' };
  int64_t l_dim_ids_out[ 5 ]   = { 'a', 'b', 'c', 'd', 'e' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB,
                          5,
                          2,
                          5,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 0 );
  REQUIRE( l_dim_ids_mb.size() == 2 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );

  REQUIRE( l_dim_ids_mb[ 0 ] == 'd' );
  REQUIRE( l_dim_ids_mb[ 1 ] == 'e' );
  REQUIRE( l_dim_ids_nb[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_kb[ 0 ] == 'f' );
}

TEST_CASE( "Blocking of BGEMM. Left: kb cb mb, right: nb cb kb, out: nb mb cb.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 2 ) );

  int64_t l_dim_ids_left[ 3 ]  = { 'a', 'b', 'c' };
  int64_t l_dim_ids_right[ 3 ] = { 'a', 'd', 'b' };
  int64_t l_dim_ids_out[ 3 ]   = { 'd', 'c', 'a' };

  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.blocking( einsum_ir::primblo_t::LEFT_X_CB_KB_MB_RIGHT_X_CB_NB_KB_OUT_NB_X_MB_CB,
                          3,
                          3,
                          3,
                          l_dim_ids_left,
                          l_dim_ids_right,
                          l_dim_ids_out,
                          &l_dim_sizes,
                          nullptr,
                          nullptr,
                          nullptr,
                          &l_dim_ids_cb,
                          &l_dim_ids_mb,
                          &l_dim_ids_nb,
                          &l_dim_ids_kb );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_cb.size() == 1 );
  REQUIRE( l_dim_ids_mb.size() == 1 );
  REQUIRE( l_dim_ids_nb.size() == 1 );
  REQUIRE( l_dim_ids_kb.size() == 1 );

  REQUIRE( l_dim_ids_cb[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_mb[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_nb[ 0 ] == 'd' );
  REQUIRE( l_dim_ids_kb[ 0 ] == 'b' );
} 

TEST_CASE( "Reordering of matrix-matrix multiplication which does not require reordering.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  int64_t l_num_dims_left  = 2;
  int64_t l_num_dims_right = 2;
  int64_t l_num_dims_out   = 2;

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );

  int64_t l_dim_ids_left[ 2 ]  = { 'a', 'b' };
  int64_t l_dim_ids_right[ 2 ] = { 'c', 'a' };
  int64_t l_dim_ids_out[ 2 ]   = { 'c', 'b' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         l_num_dims_left,
                         l_num_dims_right,
                         l_num_dims_out,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'b' );
  
  REQUIRE( l_dim_ids_right[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'a' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
}

TEST_CASE( "Reordering of a matrix-matrix multiplication with two K dims which have to be reordered.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 2 ) );

  int64_t l_dim_ids_left[ 3 ]  = { 'a', 'b', 'c' };
  int64_t l_dim_ids_right[ 3 ] = { 'd', 'a', 'b' };
  int64_t l_dim_ids_out[ 2 ]   = { 'd', 'c' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         3,
                         3,
                         2,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 0 ] == 'b' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'a' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'c' );
  
  REQUIRE( l_dim_ids_right[ 0 ] == 'd' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_right[ 2 ] == 'a' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'd' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'c' );
}

TEST_CASE( "Reordering of a matrix-matrix multiplication which requires a transposition of A.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  int64_t l_num_dims_left  = 2;
  int64_t l_num_dims_right = 2;
  int64_t l_num_dims_out   = 2;

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );

  int64_t l_dim_ids_left[ 2 ]  = { 'b', 'a' };
  int64_t l_dim_ids_right[ 2 ] = { 'c', 'a' };
  int64_t l_dim_ids_out[ 2 ]   = { 'c', 'b' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         l_num_dims_left,
                         l_num_dims_right,
                         l_num_dims_out,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'b' );
  
  REQUIRE( l_dim_ids_right[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'a' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
}

TEST_CASE( "Reordering of a matrix-matrix multiplication which requires a transposition of B.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  int64_t l_num_dims_left  = 2;
  int64_t l_num_dims_right = 2;
  int64_t l_num_dims_out   = 2;

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );

  int64_t l_dim_ids_left[ 2 ]  = { 'a', 'b' };
  int64_t l_dim_ids_right[ 2 ] = { 'a', 'c' };
  int64_t l_dim_ids_out[ 2 ]   = { 'c', 'b' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         l_num_dims_left,
                         l_num_dims_right,
                         l_num_dims_out,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'b' );
  
  REQUIRE( l_dim_ids_right[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'a' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
}

TEST_CASE( "Reordering of a high-dimensional contraction where the left tensor requires reordering.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  int64_t l_num_dims_left  = 5;
  int64_t l_num_dims_right = 3;
  int64_t l_num_dims_out   = 4;

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 2 ) );

  int64_t l_dim_ids_left[ 5 ]  = { 'a', 'b', 'c', 'd', 'e' };
  int64_t l_dim_ids_right[ 3 ] = { 'f', 'd', 'e' };
  int64_t l_dim_ids_out[ 4 ]   = { 'b', 'f', 'c', 'a' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         l_num_dims_left,
                         l_num_dims_right,
                         l_num_dims_out,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  // print dim ids left as string
  std::string l_dim_ids_left_str = "";
  for ( int i = 0; i < l_num_dims_left; i++ ) {
    l_dim_ids_left_str += l_dim_ids_left[ i ];
  }

  REQUIRE( l_dim_ids_left[ 0 ] == 'b' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'd' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'e' );
  REQUIRE( l_dim_ids_left[ 3 ] == 'c' );
  REQUIRE( l_dim_ids_left[ 4 ] == 'a' );
  
  REQUIRE( l_dim_ids_right[ 0 ] == 'f' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'd' );
  REQUIRE( l_dim_ids_right[ 2 ] == 'e' );


  REQUIRE( l_dim_ids_out[ 0 ] == 'b' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'f' );
  REQUIRE( l_dim_ids_out[ 2 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 3 ] == 'a' );
}

TEST_CASE( "Reordering of a high-dimensional contraction where the right tensor requires reordering.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 2, 8, 2, 8, 2, 8, 2, 8 );

  int64_t l_num_dims_left  = 5;
  int64_t l_num_dims_right = 6;
  int64_t l_num_dims_out   = 4;

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'x', 2 ) );

  int64_t l_dim_ids_left[ 5 ]  = { 'x', 'a', 'b', 'c', 'd' };
  int64_t l_dim_ids_right[ 6 ] = { 'b', 'c', 'a', 'x', 'e', 'f' };
  int64_t l_dim_ids_out[ 4 ]   = { 'f', 'x', 'e', 'd' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         l_num_dims_left,
                         l_num_dims_right,
                         l_num_dims_out,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  // print dim ids left as string
  std::string l_dim_ids_left_str = "";
  for ( int i = 0; i < l_num_dims_left; i++ ) {
    l_dim_ids_left_str += l_dim_ids_left[ i ];
  }

  REQUIRE( l_dim_ids_left[ 0 ] == 'x' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'a' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'b' );
  REQUIRE( l_dim_ids_left[ 3 ] == 'c' );
  REQUIRE( l_dim_ids_left[ 4 ] == 'd' );
  
  REQUIRE( l_dim_ids_right[ 0 ] == 'x' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'f' );
  REQUIRE( l_dim_ids_right[ 2 ] == 'e' );
  REQUIRE( l_dim_ids_right[ 3 ] == 'a' );
  REQUIRE( l_dim_ids_right[ 4 ] == 'b' );
  REQUIRE( l_dim_ids_right[ 5 ] == 'c' );


  REQUIRE( l_dim_ids_out[ 0 ] == 'f' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'x' );
  REQUIRE( l_dim_ids_out[ 2 ] == 'e' );
  REQUIRE( l_dim_ids_out[ 3 ] == 'd' );
}

TEST_CASE( "Reordering of TCCG setting #1: efbad,cf->abcde 48,36,24,36,48,36.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );

  int64_t l_dim_ids_left[ 5 ]  = { 'e', 'f', 'b', 'a', 'd' };
  int64_t l_dim_ids_right[ 2 ] = { 'c', 'f' };
  int64_t l_dim_ids_out[ 5 ]   = { 'a', 'b', 'c', 'd', 'e' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         5,
                         2,
                         5,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 4 ] == 'e' );
  REQUIRE( l_dim_ids_left[ 3 ] == 'f' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'd' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_left[ 0 ] == 'a' );

  REQUIRE( l_dim_ids_right[ 1 ] == 'f' );
  REQUIRE( l_dim_ids_right[ 0 ] == 'c' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_out[ 2 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 3 ] == 'd' );
  REQUIRE( l_dim_ids_out[ 4 ] == 'e' );
}

TEST_CASE( "Reordering of TCCG setting #1 with batch: efbadx,cfx->abcdex 48,36,24,36,48,36,16", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'x', 16 ) );

  int64_t l_dim_ids_left[ 6 ]  = { 'e', 'f', 'b', 'a', 'd', 'x' };
  int64_t l_dim_ids_right[ 3 ] = { 'c', 'f', 'x' };
  int64_t l_dim_ids_out[ 6 ]   = { 'a', 'b', 'c', 'd', 'e', 'x' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_CB_RIGHT_BC_BN_BK_BJ_NB_KB_CB_OUT_NATIVE,
                         6,
                         3,
                         6,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 5 ] == 'x' );
  REQUIRE( l_dim_ids_left[ 4 ] == 'e' );
  REQUIRE( l_dim_ids_left[ 3 ] == 'f' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'd' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_left[ 0 ] == 'a' );

  REQUIRE( l_dim_ids_right[ 2 ] == 'x' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'f' );
  REQUIRE( l_dim_ids_right[ 0 ] == 'c' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_out[ 2 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 3 ] == 'd' );
  REQUIRE( l_dim_ids_out[ 4 ] == 'e' );
  REQUIRE( l_dim_ids_out[ 5 ] == 'x' );
}

TEST_CASE( "Reordering of TCCG setting #1 with batch: efbadx,cfx->abcdxe 48,36,24,36,48,36,16" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'x', 16 ) );

  int64_t l_dim_ids_left[ 6 ]  = { 'e', 'f', 'b', 'a', 'd', 'x' };
  int64_t l_dim_ids_right[ 3 ] = { 'c', 'f', 'x' };
  int64_t l_dim_ids_out[ 6 ]   = { 'a', 'b', 'c', 'd', 'x', 'e' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         6,
                         3,
                         6,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 5 ] == 'e' );
  REQUIRE( l_dim_ids_left[ 4 ] == 'f' );
  REQUIRE( l_dim_ids_left[ 3 ] == 'd' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'b' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'a' );
  REQUIRE( l_dim_ids_left[ 0 ] == 'x' );

  REQUIRE( l_dim_ids_right[ 2 ] == 'f' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'c' );
  REQUIRE( l_dim_ids_right[ 0 ] == 'x' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_out[ 2 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 3 ] == 'd' );
  REQUIRE( l_dim_ids_out[ 4 ] == 'x' );
  REQUIRE( l_dim_ids_out[ 5 ] == 'e' );
}

TEST_CASE( "Reordering of TCCG setting #1 with batch: efbadx,cfx->abcxde 48,36,24,36,48,36,16" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 48 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 36 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'x', 16 ) );

  int64_t l_dim_ids_left[ 6 ]  = { 'e', 'f', 'b', 'a', 'd', 'x' };
  int64_t l_dim_ids_right[ 3 ] = { 'c', 'f', 'x' };
  int64_t l_dim_ids_out[ 6 ]   = { 'a', 'b', 'c', 'x', 'd', 'e' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         6,
                         3,
                         6,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 5 ] == 'e' );
  REQUIRE( l_dim_ids_left[ 4 ] == 'f' );
  REQUIRE( l_dim_ids_left[ 3 ] == 'd' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'b' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'a' );
  REQUIRE( l_dim_ids_left[ 0 ] == 'x' );

  REQUIRE( l_dim_ids_right[ 2 ] == 'f' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'c' );
  REQUIRE( l_dim_ids_right[ 0 ] == 'x' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_out[ 2 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 3 ] == 'x' );
  REQUIRE( l_dim_ids_out[ 4 ] == 'd' );
  REQUIRE( l_dim_ids_out[ 5 ] == 'e' );
}

TEST_CASE( "Reordering of TCCG setting #8 (swapped): gfbc,dega->abcdef 24,20,20,24,20,20,24.", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 20 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 20 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 24 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 20 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 20 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'g', 24 ) );

  int64_t l_dim_ids_left[ 4 ]  = { 'g', 'f', 'b', 'c' };
  int64_t l_dim_ids_right[ 4 ] = { 'd', 'e', 'g', 'a' };
  int64_t l_dim_ids_out[ 6 ]   = { 'a', 'b', 'c', 'd', 'e', 'f' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE,
                         4,
                         4,
                         6,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 3 ] == 'f' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'g' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'c' );
  REQUIRE( l_dim_ids_left[ 0 ] == 'b' );

  REQUIRE( l_dim_ids_right[ 3 ] == 'g' );
  REQUIRE( l_dim_ids_right[ 2 ] == 'e' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'd' );
  REQUIRE( l_dim_ids_right[ 0 ] == 'a' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'a' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_out[ 2 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 3 ] == 'd' );
  REQUIRE( l_dim_ids_out[ 4 ] == 'e' );
  REQUIRE( l_dim_ids_out[ 5 ] == 'f' );
}

TEST_CASE( "Reordering of a packed GEMM for the BLAS interface", "[binary_primitives]" ) {
  einsum_ir::backend::BinaryPrimitives l_bpr;

  l_bpr.init( 4,    8,   // C
              32, 128,   // M
              16,  64,   // N
              64, 512 ); // K

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a',  4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b',  4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c',  4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'x', 16 ) );

  int64_t l_dim_ids_left[ 3 ]  = { 'a', 'b', 'x' };
  int64_t l_dim_ids_right[ 3 ] = { 'c', 'a', 'x' };
  int64_t l_dim_ids_out[ 3 ]   = { 'c', 'b', 'x' };

  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;

  l_err = l_bpr.reorder( einsum_ir::tenord_t::LEFT_BC_BM_BK_BI_CB_KB_MB_RIGHT_BC_BN_BK_BJ_CB_NB_KB_OUT_NATIVE,
                         3,
                         3,
                         3,
                         &l_dim_sizes,
                         l_dim_ids_left,
                         l_dim_ids_right,
                         l_dim_ids_out );

  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_dim_ids_left[ 0 ] == 'x' );
  REQUIRE( l_dim_ids_left[ 1 ] == 'a' );
  REQUIRE( l_dim_ids_left[ 2 ] == 'b' );

  REQUIRE( l_dim_ids_right[ 0 ] == 'x' );
  REQUIRE( l_dim_ids_right[ 1 ] == 'c' );
  REQUIRE( l_dim_ids_right[ 2 ] == 'a' );

  REQUIRE( l_dim_ids_out[ 0 ] == 'c' );
  REQUIRE( l_dim_ids_out[ 1 ] == 'b' );
  REQUIRE( l_dim_ids_out[ 2 ] == 'x' );
}