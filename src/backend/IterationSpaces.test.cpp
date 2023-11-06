#include "catch.hpp"
#include "IterationSpaces.h"

/**
 * Dimension hook which realizes a convolution with zero padding.
 *
 * @param i_primary_first_global primary dimension's first entries (global view).
 * @param i_primary_first_global primary dimension's sizes (global view).
 * @param i_primary_iter current iteration of the primary dimension.
 * @param i_secondary_first_global secondary dimension's first entries (global view).
 * @param i_secondary_first_global secondary dimension's sizes (global view).
 * @param o_secondary_first will be set to first entry of derived loop.
 * @param o_secondary_size will be set to size of derived loop.
 **/
void hook_conv_pad( int64_t const   i_primary_first_global,
                    int64_t const   i_primary_size_global,
                    int64_t const   i_primary_iter,
                    int64_t const   i_secondary_first_global,
                    int64_t const   i_secondary_size_global,
                    int64_t       * o_secondary_first,
                    int64_t       * o_secondary_size ) {
  int64_t l_num_pad = i_secondary_size_global / 2;

  int64_t l_num_left  = (i_primary_iter - i_primary_first_global > l_num_pad) ? l_num_pad : i_primary_iter;

  int64_t l_rem_iters = i_primary_first_global + i_primary_size_global - i_primary_iter;
  int64_t l_num_center_right = (l_rem_iters > l_num_pad + 1) ? l_num_pad + 1 : l_rem_iters;

  *o_secondary_first = i_secondary_first_global - l_num_left;
  *o_secondary_size = l_num_left + l_num_center_right;
}

TEST_CASE( "Iteration spaces with #tasks matching the sizes of the collapsed loops.", "[iteration_spaces]" ) {
  einsum_ir::backend::IterationSpaces l_iter_spaces;

  int64_t l_global_sizes[5] = { 2, 2, 3, 64, 12 };

  l_iter_spaces.init( 5,
                      3,
                      nullptr,
                      l_global_sizes,
                      12 );

  einsum_ir::err_t l_err = l_iter_spaces.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_iter_spaces.num_collapsed() == 3 );
  REQUIRE( l_iter_spaces.num_tasks() == 12 );

  int64_t const * l_firsts = nullptr;
  int64_t const * l_sizes = nullptr;

  // 1st task
  l_firsts = l_iter_spaces.firsts( 0 );
  l_sizes  = l_iter_spaces.sizes( 0 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 2nd task
  l_firsts = l_iter_spaces.firsts( 1 );
  l_sizes  = l_iter_spaces.sizes(  1 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  1 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 3rd task
  l_firsts = l_iter_spaces.firsts( 2 );
  l_sizes  = l_iter_spaces.sizes(  2 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  2 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 4th task
  l_firsts = l_iter_spaces.firsts( 3 );
  l_sizes  = l_iter_spaces.sizes(  3 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 5th task
  l_firsts = l_iter_spaces.firsts( 4 );
  l_sizes  = l_iter_spaces.sizes(  4 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  1 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 6th task
  l_firsts = l_iter_spaces.firsts( 5 );
  l_sizes  = l_iter_spaces.sizes(  5 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  2 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 7th task
  l_firsts = l_iter_spaces.firsts( 6 );
  l_sizes  = l_iter_spaces.sizes(  6 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 8th task
  l_firsts = l_iter_spaces.firsts( 7 );
  l_sizes  = l_iter_spaces.sizes(  7 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  1 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );


  // 9th task
  l_firsts = l_iter_spaces.firsts( 8 );
  l_sizes  = l_iter_spaces.sizes(  8 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  2 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 10th task
  l_firsts = l_iter_spaces.firsts( 9 );
  l_sizes  = l_iter_spaces.sizes(  9 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 11th task
  l_firsts = l_iter_spaces.firsts( 10 );
  l_sizes  = l_iter_spaces.sizes(  10 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  1 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );

  // 12th task
  l_firsts = l_iter_spaces.firsts( 11 );
  l_sizes  = l_iter_spaces.sizes(  11 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  2 );
  REQUIRE( l_firsts[3] ==  0 );
  REQUIRE( l_firsts[4] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  ==  1 );
  REQUIRE( l_sizes[3]  == 64 );
  REQUIRE( l_sizes[4]  == 12 );
}

TEST_CASE( "Iteration spaces with #tasks not matching the sizes of the collapsed loops but sufficient parallelism.", "[iteration_spaces]" ) {
  einsum_ir::backend::IterationSpaces l_iter_spaces;

  int64_t l_global_sizes[3] = { 2, 4, 32 };

  l_iter_spaces.init( 3,
                      3,
                      nullptr,
                      l_global_sizes,
                      6 );

  einsum_ir::err_t l_err = l_iter_spaces.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_iter_spaces.num_collapsed() == 2 );
  REQUIRE( l_iter_spaces.num_tasks() == 8 );

  int64_t const * l_firsts = nullptr;
  int64_t const * l_sizes = nullptr;

  // 1st task
  l_firsts = l_iter_spaces.firsts( 0 );
  l_sizes  = l_iter_spaces.sizes( 0 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 2nd task
  l_firsts = l_iter_spaces.firsts( 1 );
  l_sizes  = l_iter_spaces.sizes( 1 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 3rd task
  l_firsts = l_iter_spaces.firsts( 2 );
  l_sizes  = l_iter_spaces.sizes( 2 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  2 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 4th task
  l_firsts = l_iter_spaces.firsts( 3 );
  l_sizes  = l_iter_spaces.sizes( 3 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  3 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 5th task
  l_firsts = l_iter_spaces.firsts( 4 );
  l_sizes  = l_iter_spaces.sizes( 4 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 6th task
  l_firsts = l_iter_spaces.firsts( 5 );
  l_sizes  = l_iter_spaces.sizes( 5 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 7th task
  l_firsts = l_iter_spaces.firsts( 6 );
  l_sizes  = l_iter_spaces.sizes( 6 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  2 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 8th task
  l_firsts = l_iter_spaces.firsts( 7 );
  l_sizes  = l_iter_spaces.sizes( 7 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  3 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );
}

TEST_CASE( "Iteration spaces with too many #tasks for the given parallelizable loops.", "[iteration_spaces]" ) {
  einsum_ir::backend::IterationSpaces l_iter_spaces;

  int64_t l_global_sizes[3] = { 2, 3, 32 };

  l_iter_spaces.init( 3,
                      2,
                      nullptr,
                      l_global_sizes,
                      10 );

  einsum_ir::err_t l_err = l_iter_spaces.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_iter_spaces.num_collapsed() == 2 );
  REQUIRE( l_iter_spaces.num_tasks() == 6 );

  int64_t const * l_firsts = nullptr;
  int64_t const * l_sizes = nullptr;

  // 1st task
  l_firsts = l_iter_spaces.firsts( 0 );
  l_sizes  = l_iter_spaces.sizes( 0 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 2nd task
  l_firsts = l_iter_spaces.firsts( 1 );
  l_sizes  = l_iter_spaces.sizes( 1 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 3rd task
  l_firsts = l_iter_spaces.firsts( 2 );
  l_sizes  = l_iter_spaces.sizes( 2 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  2 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 4th task
  l_firsts = l_iter_spaces.firsts( 3 );
  l_sizes  = l_iter_spaces.sizes( 3 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 5th task
  l_firsts = l_iter_spaces.firsts( 4 );
  l_sizes  = l_iter_spaces.sizes( 4 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  1 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );

  // 6th task
  l_firsts = l_iter_spaces.firsts( 5 );
  l_sizes  = l_iter_spaces.sizes( 5 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  2 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  1 );
  REQUIRE( l_sizes[2]  == 32 );
}

TEST_CASE( "Single parallelizable loop.", "[iteration_spaces]" ) {
  einsum_ir::backend::IterationSpaces l_iter_spaces;

  int64_t l_global_sizes[3] = { 5, 2, 3 };

  l_iter_spaces.init( 3,
                      1,
                      nullptr,
                      l_global_sizes,
                      3 );

  einsum_ir::err_t l_err = l_iter_spaces.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_iter_spaces.num_collapsed() == 1 );
  REQUIRE( l_iter_spaces.num_tasks() == 5 );

  int64_t const * l_firsts = nullptr;
  int64_t const * l_sizes = nullptr;

  // 1st task
  l_firsts = l_iter_spaces.firsts( 0 );
  l_sizes  = l_iter_spaces.sizes( 0 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  2 );
  REQUIRE( l_sizes[2]  ==  3 );

  // 2nd task
  l_firsts = l_iter_spaces.firsts( 1 );
  l_sizes  = l_iter_spaces.sizes( 1 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  2 );
  REQUIRE( l_sizes[2]  ==  3 );

  // 3rd task
  l_firsts = l_iter_spaces.firsts( 2 );
  l_sizes  = l_iter_spaces.sizes( 2 );

  REQUIRE( l_firsts[0] ==  2 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  2 );
  REQUIRE( l_sizes[2]  ==  3 );

  // 4th task
  l_firsts = l_iter_spaces.firsts( 3 );
  l_sizes  = l_iter_spaces.sizes( 3 );

  REQUIRE( l_firsts[0] ==  3 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  2 );
  REQUIRE( l_sizes[2]  ==  3 );

  // 5th task
  l_firsts = l_iter_spaces.firsts( 4 );
  l_sizes  = l_iter_spaces.sizes( 4 );

  REQUIRE( l_firsts[0] ==  4 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  ==  2 );
  REQUIRE( l_sizes[2]  ==  3 );
}

TEST_CASE( "Sequential test case.", "[iteration_spaces]" ) {
  einsum_ir::backend::IterationSpaces l_iter_spaces;

  int64_t l_global_sizes[3] = { 17, 2, 3 };

  l_iter_spaces.init( 3,
                      0,
                      nullptr,
                      l_global_sizes,
                      3 );

  einsum_ir::err_t l_err = l_iter_spaces.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_iter_spaces.num_collapsed() == 0 );
  REQUIRE( l_iter_spaces.num_tasks() == 1 );

  int64_t const * l_firsts = nullptr;
  int64_t const * l_sizes = nullptr;

  // 1st task
  l_firsts = l_iter_spaces.firsts( 0 );
  l_sizes  = l_iter_spaces.sizes( 0 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );

  REQUIRE( l_sizes[0]  ==  17 );
  REQUIRE( l_sizes[1]  ==  2 );
  REQUIRE( l_sizes[2]  ==  3 );
}

TEST_CASE( "Iteration space with a hook.", "[iteration_spaces]" ) {
  einsum_ir::backend::IterationSpaces l_iter_spaces;

  int64_t l_global_sizes[4] = { 3, 32, 3, 7 };

  l_iter_spaces.init( 4,
                      2,
                      nullptr,
                      l_global_sizes,
                      3 );

  einsum_ir::err_t l_err = l_iter_spaces.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  REQUIRE( l_iter_spaces.num_collapsed() == 1 );
  REQUIRE( l_iter_spaces.num_tasks() == 3 );

  // add hook
  l_iter_spaces.init_dim_hook( 1,
                               3,
                               hook_conv_pad );

  int64_t const * l_firsts = nullptr;
  int64_t const * l_sizes = nullptr;

  /*
   * check results before executing the hook
   */
  // 1st task
  l_firsts = l_iter_spaces.firsts( 0 );
  l_sizes  = l_iter_spaces.sizes( 0 );

  REQUIRE( l_firsts[0] ==  0 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  7 );

  // 2nd task
  l_firsts = l_iter_spaces.firsts( 1 );
  l_sizes  = l_iter_spaces.sizes( 1 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  7 );

  // 3rd task
  l_firsts = l_iter_spaces.firsts( 2 );
  l_sizes  = l_iter_spaces.sizes( 2 );

  REQUIRE( l_firsts[0] ==  2 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  7 );

  /**
   * Execute hook at domain boundary of assumed convolution operator with kernel size 7
   **/
  l_iter_spaces.eval_dim_hook( 1,
                               1,
                               0 );

  l_firsts = l_iter_spaces.firsts( 1 );
  l_sizes  = l_iter_spaces.sizes( 1 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] ==  0 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  4 );

  l_iter_spaces.eval_dim_hook( 1,
                               1,
                               1 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] == -1 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  5 );

  l_iter_spaces.eval_dim_hook( 1,
                               1,
                               2 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] == -2 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  6 );

  l_iter_spaces.eval_dim_hook( 1,
                               1,
                               3 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] == -3 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  7 );

  l_iter_spaces.eval_dim_hook( 1,
                               1,
                               4 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] == -3 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  7 );

  for( int64_t l_it = 5; l_it < 29; l_it++ ) {
    l_iter_spaces.eval_dim_hook(  1,
                                  1,
                                 15 );

    REQUIRE( l_firsts[0] ==  1 );
    REQUIRE( l_firsts[1] ==  0 );
    REQUIRE( l_firsts[2] ==  0 );
    REQUIRE( l_firsts[3] == -3 );

    REQUIRE( l_sizes[0]  ==  1 );
    REQUIRE( l_sizes[1]  == 32 );
    REQUIRE( l_sizes[2]  ==  3 );
    REQUIRE( l_sizes[3]  ==  7 );
  }

  l_iter_spaces.eval_dim_hook(  1,
                                1,
                               29 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] == -3 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  6 );

  l_iter_spaces.eval_dim_hook(  1,
                                1,
                               30 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] == -3 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  5 );

  l_iter_spaces.eval_dim_hook(  1,
                                1,
                               31 );

  REQUIRE( l_firsts[0] ==  1 );
  REQUIRE( l_firsts[1] ==  0 );
  REQUIRE( l_firsts[2] ==  0 );
  REQUIRE( l_firsts[3] == -3 );

  REQUIRE( l_sizes[0]  ==  1 );
  REQUIRE( l_sizes[1]  == 32 );
  REQUIRE( l_sizes[2]  ==  3 );
  REQUIRE( l_sizes[3]  ==  4 );
}