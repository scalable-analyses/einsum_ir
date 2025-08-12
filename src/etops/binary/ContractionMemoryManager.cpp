#include "ContractionMemoryManager.h"

#ifdef _OPENMP
#include <omp.h>
#endif

etops::binary::ContractionMemoryManager::~ContractionMemoryManager() {
  for( std::size_t l_id = 0; l_id < m_thread_memory.size(); l_id++ ){
    if( m_thread_memory[l_id] != nullptr ){
      delete [] (char *) m_thread_memory[l_id];
    }
  }
}

void etops::binary::ContractionMemoryManager::alloc_all_memory(){
  if( m_req_thread_mem ){
    m_thread_memory.resize( m_num_threads, nullptr );
    m_aligned_thread_memory.resize(m_num_threads, nullptr);

#ifdef _OPENMP
#pragma omp parallel for num_threads(m_num_threads)
#endif
    for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ){
      //allocate memory
      char * l_ptr = new char[ m_req_thread_mem + m_alignment_line ];
      m_thread_memory[l_thread_id] = l_ptr;

      //allign data in memory
      int64_t l_align_offset = (unsigned long)l_ptr % m_alignment_line;
      l_align_offset = l_align_offset ? m_alignment_line - l_align_offset : 0;
      m_aligned_thread_memory[l_thread_id] = l_ptr + l_align_offset;

      //first touch policy
      for( int64_t l_mem_id = 0; l_mem_id <= m_req_thread_mem; l_mem_id++ ){
        m_aligned_thread_memory[l_thread_id][l_mem_id] = 0;
      }
    }
  }
}

void etops::binary::ContractionMemoryManager::reserve_thread_memory( int64_t i_size, 
                                                                     int64_t i_num_threads ){
  if( i_size > m_req_thread_mem ){
    m_req_thread_mem = i_size;
  }
  if( i_num_threads > m_num_threads ){
    m_num_threads = i_num_threads;
  }
}

char * etops::binary::ContractionMemoryManager::get_thread_memory( int64_t i_thread_id ){
  if( i_thread_id < m_num_threads ){
    return m_aligned_thread_memory[i_thread_id];
  }
  return nullptr;
}