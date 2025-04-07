#include "catch.hpp"
#include "ContractionOptimizer.h"
#include "../../constants.h"
#include <iostream>

TEST_CASE( "Simple test for Contractoin Optimizer", "[contraction_optimizer]" ) {
  //example: [c1,k1,m1],[c1,n1,k1]->[c1,n1,m1]
  //sizes:   [17,13,20],[17,47,13]->[17,47,20]
  using namespace einsum_ir;
  std::vector< backend::loop_property > l_loops = { {dim_t::M, exec_t::SEQ,  32,   8,  0, 0,   8},
                                                    {dim_t::K, exec_t::SEQ,  64, 256,  1, 0,   0},
                                                    {dim_t::M, exec_t::SEQ,   2,   1,  0, 0,   1},
                                                    {dim_t::N, exec_t::SEQ, 128,   0, 64, 0, 256},
                                                    {dim_t::M, exec_t::SEQ,   4,   2,  0, 0,   2}};


  backend::ContractionOptimizer l_opt;

  kernel_t l_kernel_main = kernel_t::BR_MADD;
  l_opt.init( &l_loops, &l_kernel_main, 1);  

  l_opt.optimize();

  

  REQUIRE( true );
}


TEST_CASE( "Simple matmul test for Contractoin Optimizer", "[contraction_optimizer1]" ) {
  //example: [c1,k1,m1],[c1,n1,k1]->[c1,n1,m1]
  //sizes:   [17,13,20],[17,47,13]->[17,47,20]
  using namespace einsum_ir;
  std::vector< backend::loop_property > l_loops = { {dim_t::N, exec_t::SEQ, 2048,    0, 2048, 0, 2048},
                                                    {dim_t::K, exec_t::SEQ, 2048, 2048,    1, 0,    0},
                                                    {dim_t::M, exec_t::SEQ, 2048,    1,    0, 0,    1}};


  backend::ContractionOptimizer l_opt;

  kernel_t l_kernel_main = kernel_t::BR_MADD;
  l_opt.init( &l_loops, &l_kernel_main, 1);   

  l_opt.optimize();

  for(int64_t l_id = 0; l_id < l_loops.size(); l_id++ ){
    std::cout << l_loops[l_id].size << std::endl;
  }

  REQUIRE( true );
}