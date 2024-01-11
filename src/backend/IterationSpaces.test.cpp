#include "catch.hpp"
#include "IterationSpaces.h"

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