#include "catch.hpp"
#include "ContractionOptimizer.h"

TEST_CASE( "Simple test for Contraction Optimizer", "[contraction_optimizer]" ) {
  using namespace einsum_ir;
  using namespace einsum_ir::binary;

  std::vector< loop_property > l_loops = { {dim_t::M, exec_t::SEQ,  32,   8,  0, 0,   8},
                                           {dim_t::K, exec_t::SEQ,  64, 256,  1, 0,   0},
                                           {dim_t::M, exec_t::SEQ,   2,   1,  0, 0,   1},
                                           {dim_t::N, exec_t::SEQ, 128,   0, 64, 0, 256},
                                           {dim_t::M, exec_t::SEQ,   4,   2,  0, 0,   2}};


  ContractionOptimizer l_opt;
  kernel_t l_kernel_main = kernel_t::MADD;

  int64_t l_size_before[] = {1,1,1,1};
  for( int64_t l_id = 0; l_id < l_loops.size(); l_id++ ){
    if( l_loops[l_id].dim_type == dim_t::C ){
      l_size_before[0] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::M ){
      l_size_before[1] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::N ){
      l_size_before[2] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::K ){
      l_size_before[3] *= l_loops[l_id].size;
    }
  }

  l_opt.init( &l_loops, &l_kernel_main, 1, 16, 64, 256 );  

  l_opt.optimize();

  //check that there are at least 3 primitive loops
  int64_t l_num_loops = l_loops.size();
  REQUIRE( l_num_loops > 3 );
  REQUIRE( l_loops[l_num_loops - 1].exec_type == exec_t::PRIM );
  REQUIRE( l_loops[l_num_loops - 2].exec_type == exec_t::PRIM );
  REQUIRE( l_loops[l_num_loops - 3].exec_type == exec_t::PRIM );

  //check that loops aren't of size 1
  REQUIRE( l_loops[l_num_loops - 1].size > 1 );
  REQUIRE( l_loops[l_num_loops - 2].size > 1 );
  REQUIRE( l_loops[l_num_loops - 3].size > 1 );

  //check that size of all loops is unchanged
  int64_t l_size_after[] = {1,1,1,1};
  for( int64_t l_id = 0; l_id < l_num_loops; l_id++ ){
    if( l_loops[l_id].dim_type == dim_t::C ){
      l_size_after[0] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::M ){
      l_size_after[1] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::N ){
      l_size_after[2] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::K ){
      l_size_after[3] *= l_loops[l_id].size;
    }
  }
  REQUIRE( l_size_before[0] == l_size_after[0] );
  REQUIRE( l_size_before[1] == l_size_after[1] );
  REQUIRE( l_size_before[2] == l_size_after[2] );
  REQUIRE( l_size_before[3] == l_size_after[3] );
}


TEST_CASE( "Matmul blocking test for Contraction Optimizer", "[contraction_optimizer]" ) {
  using namespace einsum_ir;
  using namespace einsum_ir::binary;

  std::vector< loop_property > l_loops = { {dim_t::N, exec_t::SEQ, 2048,    0, 2048, 0, 2048},
                                           {dim_t::K, exec_t::SEQ, 2048, 2048,    1, 0,    0},
                                           {dim_t::M, exec_t::SEQ, 2048,    1,    0, 0,    1}};


  ContractionOptimizer l_opt;
  kernel_t l_kernel_main = kernel_t::MADD;

  int64_t l_size_before[] = {1,1,1,1};
  for( int64_t l_id = 0; l_id < l_loops.size(); l_id++ ){
    if( l_loops[l_id].dim_type == dim_t::C ){
      l_size_before[0] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::M ){
      l_size_before[1] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::N ){
      l_size_before[2] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::K ){
      l_size_before[3] *= l_loops[l_id].size;
    }
  }

  l_opt.init( &l_loops, &l_kernel_main, 72, 16, 64, 256);  

  l_opt.optimize();

  //check that there are at least 3 primitive loops
  int64_t l_num_loops = l_loops.size();
  REQUIRE( l_num_loops > 3 );
  REQUIRE( l_loops[l_num_loops - 1].exec_type == exec_t::PRIM );
  REQUIRE( l_loops[l_num_loops - 2].exec_type == exec_t::PRIM );
  REQUIRE( l_loops[l_num_loops - 3].exec_type == exec_t::PRIM );

  //check that loops aren't of size 1
  REQUIRE( l_loops[l_num_loops - 1].size > 1 );
  REQUIRE( l_loops[l_num_loops - 2].size > 1 );
  REQUIRE( l_loops[l_num_loops - 3].size > 1 );

  //check that size of all loops is unchanged
  int64_t l_size_after[] = {1,1,1,1};
  for( int64_t l_id = 0; l_id < l_num_loops; l_id++ ){
    if( l_loops[l_id].dim_type == dim_t::C ){
      l_size_after[0] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::M ){
      l_size_after[1] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::N ){
      l_size_after[2] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::K ){
      l_size_after[3] *= l_loops[l_id].size;
    }
  }
  REQUIRE( l_size_before[0] == l_size_after[0] );
  REQUIRE( l_size_before[1] == l_size_after[1] );
  REQUIRE( l_size_before[2] == l_size_after[2] );
  REQUIRE( l_size_before[3] == l_size_after[3] );
}

TEST_CASE( "Test of Contraction Optimizer for transposed kernel", "[contraction_optimizer]" ) {
  using namespace einsum_ir;
  using namespace einsum_ir::binary;

  std::vector< loop_property > l_loops = { {dim_t::K, exec_t::SEQ,   64, 4096,   64, 0,    0},
                                           {dim_t::N, exec_t::SEQ,   64,    0,    1, 0,   64},
                                           {dim_t::K, exec_t::SEQ,   64,    1, 4096, 0,    0},
                                           {dim_t::M, exec_t::SEQ,   64,   64,    0, 0,    1}};


  ContractionOptimizer l_opt;
  kernel_t l_kernel_main = kernel_t::MADD;

  int64_t l_size_before[] = {1,1,1,1};
  for( int64_t l_id = 0; l_id < l_loops.size(); l_id++ ){
    if( l_loops[l_id].dim_type == dim_t::C ){
      l_size_before[0] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::M ){
      l_size_before[1] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::N ){
      l_size_before[2] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::K ){
      l_size_before[3] *= l_loops[l_id].size;
    }
  }

  l_opt.init( &l_loops, &l_kernel_main, 72, 16, 64, 256);  

  l_opt.optimize();

  //check that there are at least 3 primitive loops
  int64_t l_num_loops = l_loops.size();
  REQUIRE( l_num_loops > 3 );
  REQUIRE( l_loops[l_num_loops - 1].exec_type == exec_t::PRIM );
  REQUIRE( l_loops[l_num_loops - 2].exec_type == exec_t::PRIM );
  REQUIRE( l_loops[l_num_loops - 3].exec_type == exec_t::PRIM );

  //check that loops aren't of size 1
  REQUIRE( l_loops[l_num_loops - 1].size > 1 );
  REQUIRE( l_loops[l_num_loops - 2].size > 1 );
  REQUIRE( l_loops[l_num_loops - 3].size > 1 );

  //check that size of all loops is unchanged
  int64_t l_size_after[] = {1,1,1,1};
  for( int64_t l_id = 0; l_id < l_num_loops; l_id++ ){
    if( l_loops[l_id].dim_type == dim_t::C ){
      l_size_after[0] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::M ){
      l_size_after[1] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::N ){
      l_size_after[2] *= l_loops[l_id].size;
    }
    if( l_loops[l_id].dim_type == dim_t::K ){
      l_size_after[3] *= l_loops[l_id].size;
    }
  }
  REQUIRE( l_size_before[0] == l_size_after[0] );
  REQUIRE( l_size_before[1] == l_size_after[1] );
  REQUIRE( l_size_before[2] == l_size_after[2] );
  REQUIRE( l_size_before[3] == l_size_after[3] );
}