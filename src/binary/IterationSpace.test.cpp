#include "catch.hpp"
#include "IterationSpace.h"

#include <iostream>
TEST_CASE( "Threading test for Iteration Space with SFC+OMP, 3 threads and broadcast of scalar value.", "[iter_space_sfc]" ) {
  //example: [c1,m2,k2,k1,m1],[c1,n2,k2,n1,k1]->[c1,n2,m2,n1,m1]
  //sizes:   [ 2, 3, 5, 2, 2],[ 2, 4, 5, 2, 2]->[ 2, 4, 3, 2, 2]
  //strides: [60,20, 4, 2, 1],[80,20, 4, 2, 1]->[48,12, 4, 2, 1]
  using namespace einsum_ir;
  using namespace einsum_ir::binary;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C, 
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K, 
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::OMP, 
                                             exec_t::SFC, 
                                             exec_t::SFC, 
                                             exec_t::SEQ, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                c1 n2,m2,k2,m1,n1,k1
  std::vector< int64_t > l_loop_sizes           = {  2, 3, 4, 5, 2, 2, 2};
  std::vector< int64_t > l_loop_strides_left    = { 60,20, 0, 4, 2, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = { 80, 0,20, 4, 0, 2, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0, 1, 1, 1};
  std::vector< int64_t > l_loop_strides_out     = { 48, 4,12, 0, 2, 1, 0};

  IterationSpace l_iter;

  int64_t l_num_threads = 3;

  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               &l_loop_strides_left,
               &l_loop_strides_right,
               &l_loop_strides_out_aux,
               &l_loop_strides_out,
               l_num_threads );     
                
  err_t l_err = l_iter.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  //check that all task are distributed
  int64_t l_num_tasks = 0;
  for(int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++ ){
    l_num_tasks += l_iter.getNumTasks(l_thread_id);
  } 
  REQUIRE( l_num_tasks  == 2*3*4);

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  const char * l_ptr_left    = 0;
  const char * l_ptr_right   = 0;
  const char * l_ptr_out_aux = 0;
  char       * l_ptr_out     = 0;
  for( int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++){
    int64_t l_offset_left    = 0;
    int64_t l_offset_right   = 0;
    int64_t l_offset_out_aux = 0;
    int64_t l_offset_out     = 0;
    l_iter.getInitialOffsets(l_thread_id, l_offset_left, l_offset_right, l_offset_out_aux, l_offset_out);

    char * l_ptr_rec_left    = 0;
    char * l_ptr_rec_right   = 0;
    char * l_ptr_rec_out_aux = 0;
    char * l_ptr_rec_out     = 0;
    l_ptr_rec_left    += l_offset_left;
    l_ptr_rec_right   += l_offset_right;
    l_ptr_rec_out_aux += l_offset_out_aux;
    l_ptr_rec_out     += l_offset_out; 

    REQUIRE( l_ptr_left    == l_ptr_rec_left    );
    REQUIRE( l_ptr_right   == l_ptr_rec_right   );
    REQUIRE( l_ptr_out_aux == l_ptr_rec_out_aux );
    REQUIRE( l_ptr_out     == l_ptr_rec_out     );

    int64_t l_size = l_iter.getNumTasks(l_thread_id);
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      l_iter.addMovementOffsets(l_thread_id, l_id, &l_ptr_left, &l_ptr_right, &l_ptr_out_aux, &l_ptr_out);
    }
  }
}


TEST_CASE( "Threading test for OMP only Iteration Space, 5 threads and auxiliary bias tensor.", "[iter_space_sfc]" ) {
  //example: [c1,m2,k2,k1,m1],[c1,n2,k2,n1,k1]->[c1,n2,m2,n1,m1]
  //sizes:   [ 2, 3, 5, 2, 2],[ 2, 4, 5, 2, 2]->[ 2, 4, 3, 2, 2]
  //strides: [60,20, 4, 2, 1],[80,20, 4, 2, 1]->[48,12, 4, 2, 1]
  using namespace einsum_ir;
  using namespace einsum_ir::binary;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C, 
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K, 
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::OMP, 
                                             exec_t::OMP, 
                                             exec_t::OMP, 
                                             exec_t::SEQ, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                c1 m2,n2,k2,m1,n1,k1
  std::vector< int64_t > l_loop_sizes           = {  2, 3, 4, 5, 2, 2, 2};
  std::vector< int64_t > l_loop_strides_left    = { 60,20, 0, 4, 2, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = { 80, 0,20, 4, 0, 2, 1};
  std::vector< int64_t > l_loop_strides_out_aux = { 48, 4,12, 0, 2, 1, 0};
  std::vector< int64_t > l_loop_strides_out     = { 48, 4,12, 0, 2, 1, 0};

  IterationSpace l_iter;

  int64_t l_num_threads = 5;

  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               &l_loop_strides_left,
               &l_loop_strides_right,
               &l_loop_strides_out_aux,
               &l_loop_strides_out,
               5 );     
                
  err_t l_err = l_iter.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  //check that all task are distributed
  int64_t l_num_tasks = 0;
  for(int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++ ){
    l_num_tasks += l_iter.getNumTasks(l_thread_id);
  } 
  REQUIRE( l_num_tasks  == 2*3*4);

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  const char * l_ptr_left    = 0;
  const char * l_ptr_right   = 0;
  const char * l_ptr_out_aux = 0;
  char       * l_ptr_out     = 0;
  for( int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++){
    int64_t l_offset_left    = 0;
    int64_t l_offset_right   = 0;
    int64_t l_offset_out_aux = 0;
    int64_t l_offset_out     = 0;
    l_iter.getInitialOffsets(l_thread_id, l_offset_left, l_offset_right, l_offset_out_aux, l_offset_out);

    char * l_ptr_rec_left    = 0;
    char * l_ptr_rec_right   = 0;
    char * l_ptr_rec_out_aux = 0;
    char * l_ptr_rec_out     = 0;
    l_ptr_rec_left    += l_offset_left;
    l_ptr_rec_right   += l_offset_right;
    l_ptr_rec_out_aux += l_offset_out_aux;
    l_ptr_rec_out     += l_offset_out; 

    REQUIRE( l_ptr_left    == l_ptr_rec_left    );
    REQUIRE( l_ptr_right   == l_ptr_rec_right   );
    REQUIRE( l_ptr_out_aux == l_ptr_rec_out_aux );
    REQUIRE( l_ptr_out     == l_ptr_rec_out     );

    int64_t l_size = l_iter.getNumTasks(l_thread_id);
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      l_iter.addMovementOffsets(l_thread_id, l_id, &l_ptr_left, &l_ptr_right, &l_ptr_out_aux, &l_ptr_out);
    }
  }
}


TEST_CASE( "Threading test for SFC only Iteration Space with 1 thread and no bias tensor.", "[iter_space_sfc]" ) {
  //example: [m2,k2,k1,m1],[n3,n2,k2,n1,k1]->[n3,n2,m2,n1,m1]
  //sizes:   [ 3, 5, 2, 2],[ 2, 4, 5, 2, 2]->[ 2, 4, 3, 2, 2]
  //strides: [20, 4, 2, 1],[80,20, 4, 2, 1]->[48,12, 4, 2, 1]
  using namespace einsum_ir;
  using namespace einsum_ir::binary;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::N, 
                                             dim_t::N, 
                                             dim_t::K, 
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SFC, 
                                             exec_t::SFC, 
                                             exec_t::SFC, 
                                             exec_t::SEQ, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                m2,n3 n2,k2,m1,n1,k1
  std::vector< int64_t > l_loop_sizes           = {  3, 2, 4, 5, 2, 2, 2};
  std::vector< int64_t > l_loop_strides_left    = { 20, 0, 0, 4, 2, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = {  0,80,20, 4, 0, 2, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0, 0, 0, 0};
  std::vector< int64_t > l_loop_strides_out     = {  4,48,12, 0, 2, 1, 0};

  IterationSpace l_iter;

  int64_t l_num_threads = 1;

  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               &l_loop_strides_left,
               &l_loop_strides_right,
               &l_loop_strides_out_aux,
               &l_loop_strides_out,
               l_num_threads );     
                
  err_t l_err = l_iter.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  //check that all task are distributed
  int64_t l_num_tasks = 0;
  for(int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++ ){
    l_num_tasks += l_iter.getNumTasks(l_thread_id);
  } 
  REQUIRE( l_num_tasks  == 2*3*4);

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  const char * l_ptr_left    = 0;
  const char * l_ptr_right   = 0;
  const char * l_ptr_out_aux = 0;
  char       * l_ptr_out     = 0;
  for( int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++){
    int64_t l_offset_left    = 0;
    int64_t l_offset_right   = 0;
    int64_t l_offset_out_aux = 0;
    int64_t l_offset_out     = 0;
    l_iter.getInitialOffsets(l_thread_id, l_offset_left, l_offset_right, l_offset_out_aux, l_offset_out);

    char * l_ptr_rec_left    = 0;
    char * l_ptr_rec_right   = 0;
    char * l_ptr_rec_out_aux = 0;
    char * l_ptr_rec_out     = 0;
    l_ptr_rec_left    += l_offset_left;
    l_ptr_rec_right   += l_offset_right;
    l_ptr_rec_out_aux += l_offset_out_aux;
    l_ptr_rec_out     += l_offset_out; 

    REQUIRE( l_ptr_left    == l_ptr_rec_left    );
    REQUIRE( l_ptr_right   == l_ptr_rec_right   );
    REQUIRE( l_ptr_out_aux == l_ptr_rec_out_aux );
    REQUIRE( l_ptr_out     == l_ptr_rec_out     );

    int64_t l_size = l_iter.getNumTasks(l_thread_id);
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      l_iter.addMovementOffsets(l_thread_id, l_id, &l_ptr_left, &l_ptr_right, &l_ptr_out_aux, &l_ptr_out);
    }
  }
}