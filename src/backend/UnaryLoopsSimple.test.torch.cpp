#include "ATen/ATen.h"
#include "catch.hpp"
#include "UnaryLoopsSimple.h"

/**
 * FP32 scalar kernel sets b = a.
 *
 * @param i_a input a.
 * @param o_b output b.
 **/
void kernel_copy_fp32( void const * i_a,
                       void       * o_b ) {
  float const * l_a = (float const *) i_a;
  float       * l_b = (float       *) o_b;

  *l_b = *l_a;
}

/**
 * FP64 scalar kernel sets b = a.
 *
 * @param i_a input a.
 * @param o_b output b.
 **/
void kernel_copy_fp64( void const * i_a,
                       void       * o_b ) {
  double const * l_a = (double const *) i_a;
  double       * l_b = (double       *) o_b;

  *l_b = *l_a;
}

TEST_CASE( "Tensor transposition using FP64 data.", "[unary_loops_simple]" ) {
  einsum_ir::backend::UnaryLoopsSimple l_unary_loops;

  int64_t l_sizes[3]       = {  3, 5, 4 };
  int64_t l_strides_in[3]  = { 20, 4, 1 };
  int64_t l_strides_out[3] = { 20, 1, 5 };
  int64_t l_num_bytes_in   = 8;
  int64_t l_num_bytes_out  = 8;

  l_unary_loops.init( 3,
                      l_sizes,
                      l_strides_in,
                      l_strides_out,
                      l_num_bytes_in,
                      l_num_bytes_out,
                      kernel_copy_fp64 );

  einsum_ir::err_t l_err = l_unary_loops.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  at::Tensor l_t0 = at::randn( {3, 5, 4},
                               at::ScalarType::Double );

  at::Tensor l_t1 = at::randn( {3, 4, 5},
                               at::ScalarType::Double );

  l_unary_loops.eval( l_t0.data_ptr(),
                      l_t1.data_ptr() );

  REQUIRE( at::equal( l_t0.permute( {0, 2, 1} ), l_t1 ) );
}

TEST_CASE( "Tensor transposition using FP32 data.", "[unary_loops_simple]" ) {
  einsum_ir::backend::UnaryLoopsSimple l_unary_loops;

                            //   0    1    2    3    4
  int64_t l_sizes[5]       = {   3,   2,   5,   4,   7 };
                            //   7,   5,   3,   4,   2 (target sizes)
                            // 120,  24,   8,   2,   1 (target strides)
  int64_t l_strides_in[5]  = { 280, 140,  28,   7,   1 };
  int64_t l_strides_out[5] = {   8,   1,  24,   2, 120 };
  int64_t l_num_bytes_in   = 4;
  int64_t l_num_bytes_out  = 4;

  l_unary_loops.init( 5,
                      l_sizes,
                      l_strides_in,
                      l_strides_out,
                      l_num_bytes_in,
                      l_num_bytes_out,
                      kernel_copy_fp32 );

  einsum_ir::err_t l_err = l_unary_loops.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  at::Tensor l_t0 = at::randn( {3, 2, 5, 4, 7 },
                               at::ScalarType::Float );

  at::Tensor l_t1 = at::randn( {7, 5, 3, 4, 2},
                               at::ScalarType::Float );

  l_unary_loops.eval( l_t0.data_ptr(),
                      l_t1.data_ptr() );

  REQUIRE( at::equal( l_t0.permute( {4, 2, 0, 3, 1} ), l_t1 ) );
}