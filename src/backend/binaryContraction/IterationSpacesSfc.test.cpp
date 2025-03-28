#include "catch.hpp"
#include "IterationSpacesSfc.h"
#include "../../constants.h"


TEST_CASE( "Threading test for Iteration Space with SFC+OMP and 3 threads.", "[iter_space_sfc]" ) {
  //example: [c1,m2,k2,k1,m1],[c1,n2,k2,n1,k1]->[c1,n2,m2,n1,m1]
  //sizes:   [ 2, 3, 5, 2, 2],[ 2, 4, 5, 2, 2]->[ 2, 4, 3, 2, 2]
  //strides: [60,20, 4, 2, 1],[80,20, 4, 2, 1]->[48,12, 4, 2, 1]
  std::vector< einsum_ir::dim_t >  l_loop_dim_type  = { einsum_ir::dim_t::C, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::K, 
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::K };
  std::vector< einsum_ir::exec_t > l_loop_exec_type = { einsum_ir::exec_t::OMP, 
                                                        einsum_ir::exec_t::SFC, 
                                                        einsum_ir::exec_t::SFC, 
                                                        einsum_ir::exec_t::SEQ, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM };

  //                                                c1 n2,m2,k2,m1,n1,k1
  std::vector< int64_t > l_loop_sizes           = {  2, 4, 3, 5, 2, 2, 2};
  std::vector< int64_t > l_loop_strides_left    = { 60, 0,20, 4, 2, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = { 80,20, 0, 4, 0, 2, 1};
  std::vector< int64_t > l_loop_strides_out     = { 48,12, 4, 0, 2, 1, 0};

  einsum_ir::backend::IterationSpacesSfc l_iter;

  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               &l_loop_strides_left,
               &l_loop_strides_right,
               &l_loop_strides_out,
               3 );     
                
  einsum_ir::err_t l_err = l_iter.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  int64_t l_num_tasks = l_iter.num_tasks();
  REQUIRE( l_num_tasks  == 2*3*4);

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  int64_t l_offset_left  = 0;
  int64_t l_offset_right = 0;
  int64_t l_offset_out   = 0;
  for( int64_t l_thread_id = 0; l_thread_id < 3; l_thread_id++){
    int64_t l_offset_rec_left = 0;
    int64_t l_offset_rec_right = 0;
    int64_t l_offset_rec_out = 0;
    l_iter.addInitialOffsets(l_thread_id, &l_offset_rec_left, &l_offset_rec_right, &l_offset_rec_out);
    
    REQUIRE( l_offset_left  == l_offset_rec_left  );
    REQUIRE( l_offset_right == l_offset_rec_right );
    REQUIRE( l_offset_out   == l_offset_rec_out   );

    int64_t l_size = l_iter.getNumTasks(l_thread_id);
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      l_iter.addMovementOffsets(l_thread_id, l_id, &l_offset_left, &l_offset_right, &l_offset_out);
    }
  }

  //check that last offset is correct
  int64_t l_offset_exp_left  = 60 + 3*0;
  int64_t l_offset_exp_right = 80 + 3*20;
  int64_t l_offset_exp_out   = 48 + 3*12;
  REQUIRE( l_offset_left  == l_offset_exp_left  );
  REQUIRE( l_offset_right == l_offset_exp_right );
  REQUIRE( l_offset_out   == l_offset_exp_out   );
}

TEST_CASE( "Threading test for OMP only Iteration Space and 5 threads.", "[iter_space_sfc]" ) {
  //example: [c1,m2,k2,k1,m1],[c1,n2,k2,n1,k1]->[c1,n2,m2,n1,m1]
  //sizes:   [ 2, 3, 5, 2, 2],[ 2, 4, 5, 2, 2]->[ 2, 4, 3, 2, 2]
  //strides: [60,20, 4, 2, 1],[80,20, 4, 2, 1]->[48,12, 4, 2, 1]
  std::vector< einsum_ir::dim_t >  l_loop_dim_type  = { einsum_ir::dim_t::C, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::K, 
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::K };
  std::vector< einsum_ir::exec_t > l_loop_exec_type = { einsum_ir::exec_t::OMP, 
                                                        einsum_ir::exec_t::OMP, 
                                                        einsum_ir::exec_t::OMP, 
                                                        einsum_ir::exec_t::SEQ, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM };

  //                                                c1 n2,m2,k2,m1,n1,k1
  std::vector< int64_t > l_loop_sizes           = {  2, 4, 3, 5, 2, 2, 2};
  std::vector< int64_t > l_loop_strides_left    = { 60, 0,20, 4, 2, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = { 80,20, 0, 4, 0, 2, 1};
  std::vector< int64_t > l_loop_strides_out     = { 48,12, 4, 0, 2, 1, 0};

  einsum_ir::backend::IterationSpacesSfc l_iter;

  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               &l_loop_strides_left,
               &l_loop_strides_right,
               &l_loop_strides_out,
               5 );     
                
  einsum_ir::err_t l_err = l_iter.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  int64_t l_num_tasks = l_iter.num_tasks();
  REQUIRE( l_num_tasks  == 2*3*4);

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  int64_t l_offset_left  = 0;
  int64_t l_offset_right = 0;
  int64_t l_offset_out   = 0;
  for( int64_t l_thread_id = 0; l_thread_id < 5; l_thread_id++){
    int64_t l_offset_rec_left = 0;
    int64_t l_offset_rec_right = 0;
    int64_t l_offset_rec_out = 0;
    l_iter.addInitialOffsets(l_thread_id, &l_offset_rec_left, &l_offset_rec_right, &l_offset_rec_out);
    
    REQUIRE( l_offset_left  == l_offset_rec_left  );
    REQUIRE( l_offset_right == l_offset_rec_right );
    REQUIRE( l_offset_out   == l_offset_rec_out   );

    int64_t l_size = l_iter.getNumTasks(l_thread_id);
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      l_iter.addMovementOffsets(l_thread_id, l_id, &l_offset_left, &l_offset_right, &l_offset_out);
    }
  }

  //check that last offset is correct
  int64_t l_offset_exp_left  = 60 + 3*0  + 2*20;
  int64_t l_offset_exp_right = 80 + 3*20 + 2*0;
  int64_t l_offset_exp_out   = 48 + 3*12 + 2*4;
  REQUIRE( l_offset_left  == l_offset_exp_left  );
  REQUIRE( l_offset_right == l_offset_exp_right );
  REQUIRE( l_offset_out   == l_offset_exp_out   );
}

TEST_CASE( "Threading test for SFC only Iteration Space with 1 threads.", "[iter_space_sfc]" ) {
  //example: [m2,k2,k1,m1],[n3,n2,k2,n1,k1]->[n3,n2,m2,n1,m1]
  //sizes:   [ 3, 5, 2, 2],[ 2, 4, 5, 2, 2]->[ 2, 4, 3, 2, 2]
  //strides: [20, 4, 2, 1],[80,20, 4, 2, 1]->[48,12, 4, 2, 1]
  std::vector< einsum_ir::dim_t >  l_loop_dim_type  = { einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::K, 
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::K };
  std::vector< einsum_ir::exec_t > l_loop_exec_type = { einsum_ir::exec_t::SFC, 
                                                        einsum_ir::exec_t::SFC, 
                                                        einsum_ir::exec_t::SFC, 
                                                        einsum_ir::exec_t::SEQ, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM };

  //                                                n3 n2,m2,k2,m1,n1,k1
  std::vector< int64_t > l_loop_sizes           = {  2, 4, 3, 5, 2, 2, 2};
  std::vector< int64_t > l_loop_strides_left    = {  0, 0,20, 4, 2, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = { 80,20, 0, 4, 0, 2, 1};
  std::vector< int64_t > l_loop_strides_out     = { 48,12, 4, 0, 2, 1, 0};

  einsum_ir::backend::IterationSpacesSfc l_iter;

  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               &l_loop_strides_left,
               &l_loop_strides_right,
               &l_loop_strides_out,
               1 );     
                
  einsum_ir::err_t l_err = l_iter.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  int64_t l_num_tasks = l_iter.num_tasks();
  REQUIRE( l_num_tasks  == 2*3*4);

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  int64_t l_offset_left  = 0;
  int64_t l_offset_right = 0;
  int64_t l_offset_out   = 0;
  for( int64_t l_thread_id = 0; l_thread_id < 1; l_thread_id++){
    int64_t l_offset_rec_left = 0;
    int64_t l_offset_rec_right = 0;
    int64_t l_offset_rec_out = 0;
    l_iter.addInitialOffsets(l_thread_id, &l_offset_rec_left, &l_offset_rec_right, &l_offset_rec_out);
    
    REQUIRE( l_offset_left  == l_offset_rec_left  );
    REQUIRE( l_offset_right == l_offset_rec_right );
    REQUIRE( l_offset_out   == l_offset_rec_out   );

    int64_t l_size = l_iter.getNumTasks(l_thread_id);
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      l_iter.addMovementOffsets(l_thread_id, l_id, &l_offset_left, &l_offset_right, &l_offset_out);
    }
  }

  //check that last offset is correct
  int64_t l_offset_exp_left  =  0 + 3*0 ;
  int64_t l_offset_exp_right = 80 + 3*20;
  int64_t l_offset_exp_out   = 48 + 3*12;
  REQUIRE( l_offset_left  == l_offset_exp_left  );
  REQUIRE( l_offset_right == l_offset_exp_right );
  REQUIRE( l_offset_out   == l_offset_exp_out   );
}