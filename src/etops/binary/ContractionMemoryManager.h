#ifndef EINSUM_IR_BINARY_CONTRACTION_MEMORY_MANAGER
#define EINSUM_IR_BINARY_CONTRACTION_MEMORY_MANAGER

#include <vector>
#include "constants.h"

namespace einsum_ir {
  namespace binary {
    class ContractionMemoryManager;
  }
}

class einsum_ir::binary::ContractionMemoryManager{
  private:
    // alignment of memory to cache lines in bytes 
    int64_t m_alignment_line = 64;

    //! vector with thread specific allocated memory
    std::vector<char *> m_thread_memory;
    //! vector with thread specific aligned memory
    std::vector<char *> m_aligned_thread_memory;

    //! required memory per thread
    int64_t m_req_thread_mem = 0;
    //! number of threads
    int64_t m_num_threads = 1;
    
  public:
    /**
     * Destructor.
     **/
    ~ContractionMemoryManager();

    /**
     * Allocates the required memory.
     **/
    void alloc_all_memory();

    /**
     * returns a pointer to requested memory
     *
     * @param i_id id of the memory request.
     * 
     * @return pointer to requested memory
     **/
    void * get_mem_ptr( int64_t i_id );

    /**
     * reserves thread specific memory for intermediate data in contractions. 
     *
     * @param i_size size of reserved memory.
     **/
    void reserve_thread_memory( int64_t i_size,
                                int64_t i_num_threads );

    /**
     * returns a pointer to thread specific memory
     *
     * @return pointer to requested memory
     **/
    char * get_thread_memory( int64_t i_thread_id );
};

#endif
