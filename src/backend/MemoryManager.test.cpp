#include "catch.hpp"
#include "MemoryManager.h"
#include <algorithm>

TEST_CASE( "A complex memory allocation test", "[memory_manager]" ) {
  //     __18_           __3x6_
  //    /     \         /      \
  //   15     30       3x5    5x6
  //  /  \    |       /   \    |
  // 12  20   30     3x4 4x5  6x5

  //Memory Manager
  einsum_ir::backend::MemoryManager l_memory;
  l_memory.increase_layer();

  //  | 12 | 20 | ... | 15 |
  l_memory.increase_layer();
  int64_t l_mem_id_1 = l_memory.request_memory(12 * 4);
  int64_t l_mem_id_2 = l_memory.request_memory(20 * 4);
  l_memory.decrease_layer();
  int64_t l_mem_id_3 = l_memory.request_memory(15 * 4);
  l_memory.remove_memory(l_mem_id_1);
  l_memory.remove_memory(l_mem_id_2);

  // | 30 | ... | 30 | 15 |
  l_memory.increase_layer();
  int64_t l_mem_id_4 = l_memory.request_memory(30 * 4);
  l_memory.decrease_layer();
  int64_t l_mem_id_5 = l_memory.request_memory(30 * 4);
  l_memory.remove_memory(l_mem_id_4);

  l_memory.decrease_layer();

  // | 18 | ... | 30 | 15 |
  int64_t l_mem_id_6 = l_memory.request_memory(18 * 4);
  l_memory.remove_memory(l_mem_id_3);
  l_memory.remove_memory(l_mem_id_5);


  //check that pointers are written to the correct memory side
  REQUIRE( l_mem_id_1 >= 0 );
  REQUIRE( l_mem_id_2 >= 0 );
  REQUIRE( l_mem_id_3 < 0 );
  REQUIRE( l_mem_id_4 >= 0 );
  REQUIRE( l_mem_id_5 < 0 );
  REQUIRE( l_mem_id_6 >= 0 );

  //check that right amount of memory gets allocated
  REQUIRE( l_memory.m_req_mem >= ( 30 + 30 + 15 ) * 4 );

  //allocate memory and check some pointer
  l_memory.alloc_all_memory();
  float * l_mem_1_ptr = (float*) l_memory.get_mem_ptr(l_mem_id_2);
  float * l_mem_2_ptr = (float*) l_memory.get_mem_ptr(l_mem_id_3);
  REQUIRE( l_mem_1_ptr != nullptr );
  REQUIRE( l_mem_2_ptr != nullptr);
}