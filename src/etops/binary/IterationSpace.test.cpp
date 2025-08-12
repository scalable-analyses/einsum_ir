#include "catch.hpp"
#include "IterationSpace.h"

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
  std::vector< int64_t > l_loop_sizes           = {  2, 3, 4, 5, 2, 2, 2 };
  std::vector< int64_t > l_loop_strides_left    = { 60,20, 0, 4, 2, 0, 1 };
  std::vector< int64_t > l_loop_strides_right   = { 80, 0,20, 4, 0, 2, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0, 1, 1, 1 };
  std::vector< int64_t > l_loop_strides_out     = { 48, 4,12, 0, 2, 1, 0 };

  int64_t l_num_threads = 3;
  IterationSpace l_iter;
  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               l_num_threads );     

  std::vector<thread_info> m_thread_infos;
  err_t l_err = l_iter.setup( l_loop_strides_left,
                              l_loop_strides_right,
                              l_loop_strides_out_aux,
                              l_loop_strides_out,
                              m_thread_infos );
  REQUIRE( l_err == err_t::SUCCESS );

  //check that all task are distributed
  int64_t l_num_tasks = 0;
  for(int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++ ){
    l_num_tasks += m_thread_infos[l_thread_id].movement_ids.size();
  } 
  REQUIRE( l_num_tasks  == 2*3*4);

  //create 4 arrays of size 1 to obtain random pointers
  char l_ptr_start_left [] = {0};
  char l_ptr_start_right [] = {0};
  char l_ptr_start_out_aux [] = {0};
  char l_ptr_start_out [] = {0};

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  const char * l_ptr_left    = l_ptr_start_left;
  const char * l_ptr_right   = l_ptr_start_right;
  const char * l_ptr_out_aux = l_ptr_start_out_aux;
  char       * l_ptr_out     = l_ptr_start_out;
  for( int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++){
    int64_t l_offset_left    = m_thread_infos[l_thread_id].offset_left;
    int64_t l_offset_right   = m_thread_infos[l_thread_id].offset_right;
    int64_t l_offset_out_aux = m_thread_infos[l_thread_id].offset_out_aux;
    int64_t l_offset_out     = m_thread_infos[l_thread_id].offset_out;

    char * l_ptr_rec_left    = l_ptr_start_left;
    char * l_ptr_rec_right   = l_ptr_start_right;
    char * l_ptr_rec_out_aux = l_ptr_start_out_aux;
    char * l_ptr_rec_out     = l_ptr_start_out;
    l_ptr_rec_left    += l_offset_left;
    l_ptr_rec_right   += l_offset_right;
    l_ptr_rec_out_aux += l_offset_out_aux;
    l_ptr_rec_out     += l_offset_out; 

    REQUIRE( (int64_t)l_ptr_left    == (int64_t)l_ptr_rec_left    );
    REQUIRE( (int64_t)l_ptr_right   == (int64_t)l_ptr_rec_right   );
    REQUIRE( (int64_t)l_ptr_out_aux == (int64_t)l_ptr_rec_out_aux );
    REQUIRE( (int64_t)l_ptr_out     == (int64_t)l_ptr_rec_out     );

    int64_t l_size = m_thread_infos[l_thread_id].movement_ids.size();
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      sfc_t l_move =  m_thread_infos[l_thread_id].movement_ids[l_id];
      sfc_t l_sign = (l_move & 1);
      int64_t l_direction  = 1 - ( (int64_t)l_sign << 1); 
      int64_t l_current_id = l_move >> 1;

      l_ptr_left    += l_direction * l_loop_strides_left[    l_current_id ];
      l_ptr_right   += l_direction * l_loop_strides_right[   l_current_id ];
      l_ptr_out_aux += l_direction * l_loop_strides_out_aux[ l_current_id];
      l_ptr_out     += l_direction * l_loop_strides_out[     l_current_id ];
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
  std::vector< int64_t > l_loop_sizes           = {  2, 3, 4, 5, 2, 2, 2 };
  std::vector< int64_t > l_loop_strides_left    = { 60,20, 0, 4, 2, 0, 1 };
  std::vector< int64_t > l_loop_strides_right   = { 80, 0,20, 4, 0, 2, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = { 48, 4,12, 0, 2, 1, 0 };
  std::vector< int64_t > l_loop_strides_out     = { 48, 4,12, 0, 2, 1, 0 };

  int64_t l_num_threads = 5;
  IterationSpace l_iter;
  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               l_num_threads );     

  std::vector<thread_info> m_thread_infos;
  err_t l_err = l_iter.setup( l_loop_strides_left,
                              l_loop_strides_right,
                              l_loop_strides_out_aux,
                              l_loop_strides_out,
                              m_thread_infos );
  REQUIRE( l_err == err_t::SUCCESS );

  //check that all task are distributed
  int64_t l_num_tasks = 0;
  for(int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++ ){
    l_num_tasks += m_thread_infos[l_thread_id].movement_ids.size();
  } 
  REQUIRE( l_num_tasks  == 2*3*4);

  //create 4 arrays of size 1 to obtain random pointers
  char l_ptr_start_left [] = {0};
  char l_ptr_start_right [] = {0};
  char l_ptr_start_out_aux [] = {0};
  char l_ptr_start_out [] = {0};

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  const char * l_ptr_left    = l_ptr_start_left;
  const char * l_ptr_right   = l_ptr_start_right;
  const char * l_ptr_out_aux = l_ptr_start_out_aux;
  char       * l_ptr_out     = l_ptr_start_out;
  for( int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++){
    int64_t l_offset_left    = m_thread_infos[l_thread_id].offset_left;
    int64_t l_offset_right   = m_thread_infos[l_thread_id].offset_right;
    int64_t l_offset_out_aux = m_thread_infos[l_thread_id].offset_out_aux;
    int64_t l_offset_out     = m_thread_infos[l_thread_id].offset_out;

    char * l_ptr_rec_left    = l_ptr_start_left;
    char * l_ptr_rec_right   = l_ptr_start_right;
    char * l_ptr_rec_out_aux = l_ptr_start_out_aux;
    char * l_ptr_rec_out     = l_ptr_start_out;
    l_ptr_rec_left    += l_offset_left;
    l_ptr_rec_right   += l_offset_right;
    l_ptr_rec_out_aux += l_offset_out_aux;
    l_ptr_rec_out     += l_offset_out; 

    REQUIRE( (int64_t)l_ptr_left    == (int64_t)l_ptr_rec_left    );
    REQUIRE( (int64_t)l_ptr_right   == (int64_t)l_ptr_rec_right   );
    REQUIRE( (int64_t)l_ptr_out_aux == (int64_t)l_ptr_rec_out_aux );
    REQUIRE( (int64_t)l_ptr_out     == (int64_t)l_ptr_rec_out     );

    int64_t l_size = m_thread_infos[l_thread_id].movement_ids.size();
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      sfc_t l_move =  m_thread_infos[l_thread_id].movement_ids[l_id];
      sfc_t l_sign = (l_move & 1);
      int64_t l_direction  = 1 - ( (int64_t)l_sign << 1); 
      int64_t l_current_id = l_move >> 1;

      l_ptr_left    += l_direction * l_loop_strides_left[    l_current_id ];
      l_ptr_right   += l_direction * l_loop_strides_right[   l_current_id ];
      l_ptr_out_aux += l_direction * l_loop_strides_out_aux[ l_current_id];
      l_ptr_out     += l_direction * l_loop_strides_out[     l_current_id ];
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
  std::vector< int64_t > l_loop_sizes           = {  3, 2, 4, 5, 2, 2, 2 };
  std::vector< int64_t > l_loop_strides_left    = { 20, 0, 0, 4, 2, 0, 1 };
  std::vector< int64_t > l_loop_strides_right   = {  0,80,20, 4, 0, 2, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = {  4,48,12, 0, 2, 1, 0 };

  int64_t l_num_threads = 1;
  IterationSpace l_iter;
  l_iter.init( &l_loop_dim_type,
               &l_loop_exec_type,
               &l_loop_sizes,
               l_num_threads );     

  std::vector<thread_info> m_thread_infos;
  err_t l_err = l_iter.setup( l_loop_strides_left,
                              l_loop_strides_right,
                              l_loop_strides_out_aux,
                              l_loop_strides_out,
                              m_thread_infos ); 
  REQUIRE( l_err == err_t::SUCCESS );

  //check that all task are distributed
  int64_t l_num_tasks = 0;
  for(int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++ ){
    l_num_tasks += m_thread_infos[l_thread_id].movement_ids.size();
  } 
  REQUIRE( l_num_tasks  == 2*3*4);

  //create 4 arrays of size 1 to obtain random pointers
  char l_ptr_start_left [] = {0};
  char l_ptr_start_right [] = {0};
  char l_ptr_start_out_aux [] = {0};
  char l_ptr_start_out [] = {0};

  //go through sfc and check that initial offsets are equal to calculated offsets with movement
  const char * l_ptr_left    = l_ptr_start_left;
  const char * l_ptr_right   = l_ptr_start_right;
  const char * l_ptr_out_aux = l_ptr_start_out_aux;
  char       * l_ptr_out     = l_ptr_start_out;
  for( int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++){
    int64_t l_offset_left    = m_thread_infos[l_thread_id].offset_left;
    int64_t l_offset_right   = m_thread_infos[l_thread_id].offset_right;
    int64_t l_offset_out_aux = m_thread_infos[l_thread_id].offset_out_aux;
    int64_t l_offset_out     = m_thread_infos[l_thread_id].offset_out;

    char * l_ptr_rec_left    = l_ptr_start_left;
    char * l_ptr_rec_right   = l_ptr_start_right;
    char * l_ptr_rec_out_aux = l_ptr_start_out_aux;
    char * l_ptr_rec_out     = l_ptr_start_out;
    l_ptr_rec_left    += l_offset_left;
    l_ptr_rec_right   += l_offset_right;
    l_ptr_rec_out_aux += l_offset_out_aux;
    l_ptr_rec_out     += l_offset_out; 

    REQUIRE( (int64_t)l_ptr_left    == (int64_t)l_ptr_rec_left    );
    REQUIRE( (int64_t)l_ptr_right   == (int64_t)l_ptr_rec_right   );
    REQUIRE( (int64_t)l_ptr_out_aux == (int64_t)l_ptr_rec_out_aux );
    REQUIRE( (int64_t)l_ptr_out     == (int64_t)l_ptr_rec_out     );

    int64_t l_size = m_thread_infos[l_thread_id].movement_ids.size();
    for(int64_t l_id = 0; l_id < l_size; l_id++){
      sfc_t l_move =  m_thread_infos[l_thread_id].movement_ids[l_id];
      sfc_t l_sign = (l_move & 1);
      int64_t l_direction  = 1 - ( (int64_t)l_sign << 1); 
      int64_t l_current_id = l_move >> 1;

      l_ptr_left    += l_direction * l_loop_strides_left[    l_current_id ];
      l_ptr_right   += l_direction * l_loop_strides_right[   l_current_id ];
      l_ptr_out_aux += l_direction * l_loop_strides_out_aux[ l_current_id];
      l_ptr_out     += l_direction * l_loop_strides_out[     l_current_id ];
    }
  }
}