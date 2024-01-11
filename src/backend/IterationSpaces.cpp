#include "IterationSpaces.h"

void einsum_ir::backend::IterationSpaces::init( int64_t         i_num_loops,
                                                int64_t         i_num_parallel,
                                                int64_t const * i_firsts,
                                                int64_t const * i_sizes,
                                                int64_t         i_num_tasks  ) {
  m_num_loops        = i_num_loops;
  m_num_parallel     = i_num_parallel;
  m_num_tasks_target = i_num_tasks;

  m_global_space.firsts.clear();
  m_global_space.sizes.clear();
  m_thread_local_spaces.clear();

  m_global_space.firsts.resize( m_num_loops );
  m_global_space.sizes.resize(  m_num_loops );

  for( int64_t l_lo = 0; l_lo < m_num_loops; l_lo++ ) {
    m_global_space.firsts[l_lo] = (i_firsts == nullptr) ? 0 : i_firsts[l_lo];
    m_global_space.sizes[l_lo]  = i_sizes[l_lo];
  }
}

einsum_ir::err_t einsum_ir::backend::IterationSpaces::compile() {
  // collapse loop dimensions until targeted number of tasks is reached
  m_num_tasks = 1;
  m_num_collapsed = 0;
  for( int64_t l_lo = 0; l_lo < m_num_parallel; l_lo++ ) {
    if( m_num_tasks >= m_num_tasks_target ) {
      break;
    }
    m_num_tasks *= m_global_space.sizes[l_lo];
    m_num_collapsed++;
  }

  // alloc and init
  m_thread_local_spaces.resize( m_num_tasks );
  for( int64_t l_ta = 0; l_ta < m_num_tasks; l_ta++ ) {
    m_thread_local_spaces[l_ta].firsts.resize( m_num_loops );
    m_thread_local_spaces[l_ta].sizes.resize(  m_num_loops );
    for( int64_t l_lo = 0; l_lo < m_num_loops; l_lo++ ) {
      m_thread_local_spaces[l_ta].firsts[l_lo] = 0;
      m_thread_local_spaces[l_ta].sizes[l_lo] = 0;
    }
  }

  // increase innermost chunk size if possible
  int64_t l_chunk_size = 1;
  int64_t l_prime[5] = { 2, 3, 5, 7, 11 }; // limited prime factorization

  bool l_chunk_size_inc = true;
  while( l_chunk_size_inc ) {
    l_chunk_size_inc = false;

    for( int64_t l_pr = 0; l_pr < 5; l_pr++ ) {
      if(    m_num_tasks % l_prime[l_pr] == 0
          && m_num_tasks / l_prime[l_pr] > m_num_tasks_target ) {
        int64_t l_chunk_size_next = l_chunk_size * l_prime[l_pr];
        if( m_global_space.sizes[ m_num_collapsed-1 ] % l_chunk_size_next == 0 ) {
          m_num_tasks = m_num_tasks / l_prime[l_pr];
          l_chunk_size = l_chunk_size_next;
          l_chunk_size_inc = true;
        }
      }
    }
  }

  /*
   * parallel iteration space of first task
   */
  for( int64_t l_lo = 0; l_lo < m_num_collapsed-1; l_lo++ ) {
    m_thread_local_spaces[0].firsts[l_lo] = m_global_space.firsts[l_lo];
    m_thread_local_spaces[0].sizes[l_lo]  = 1;
  }
  if( m_num_collapsed > 0 ) {
    m_thread_local_spaces[0].firsts[ m_num_collapsed-1 ] = m_global_space.firsts[ m_num_collapsed-1 ];
    m_thread_local_spaces[0].sizes[  m_num_collapsed-1 ] = l_chunk_size;
  }

  /*
   * local iteration spaces of other tasks
   */
  for( int64_t l_ta = 1; l_ta < m_num_tasks; l_ta++ ) {
    IterSpace & l_prev_space = m_thread_local_spaces[l_ta - 1];
    bool l_carry = false;

    // innermost parallel loop
    int64_t l_first = l_prev_space.firsts[ m_num_collapsed-1 ] + l_prev_space.sizes[ m_num_collapsed-1 ];
    int64_t l_size  = l_chunk_size;
  
    int64_t l_global_upper  = m_global_space.firsts[ m_num_collapsed - 1 ];
            l_global_upper += m_global_space.sizes[ m_num_collapsed - 1 ];

    // tasks in innermost collapsed loop are exhausted
    if( l_first == l_global_upper ) {
      l_first = 0;
      l_carry = true;
    }

    m_thread_local_spaces[l_ta].firsts[ m_num_collapsed-1 ] = l_first;
    m_thread_local_spaces[l_ta].sizes[ m_num_collapsed-1 ]  = l_size;

    // remaining parallel loops
    for( int64_t l_lo = m_num_collapsed-2; l_lo >= 0; l_lo-- ) {
      l_first = l_prev_space.firsts[l_lo];
      l_size  = 1;

      if( l_carry ) {
        l_first = l_prev_space.firsts[l_lo] + l_prev_space.sizes[l_lo];

        if( l_first == m_global_space.sizes[l_lo] ) {
          if( l_lo == 0 ) {
            // iteration space exhausted
            return err_t::COMPILATION_FAILED;
          }
          else {
            l_first = 0;
          }
        }
        else {
          l_carry = false;
        }
      }

      m_thread_local_spaces[l_ta].firsts[l_lo] = l_first;
      m_thread_local_spaces[l_ta].sizes[l_lo]  = l_size;
    }
  }

  /*
   * sequential loops
   */
  for( int64_t l_ta = 0; l_ta < m_num_tasks; l_ta++ ) {
    for( int64_t l_lo = m_num_collapsed; l_lo < m_num_loops; l_lo++ ) {
      m_thread_local_spaces[l_ta].firsts[l_lo] = m_global_space.firsts[l_lo];
      m_thread_local_spaces[l_ta].sizes[l_lo]  = m_global_space.sizes[l_lo];
    }
  }

  m_compiled = true;
  return err_t::SUCCESS;
}

int64_t einsum_ir::backend::IterationSpaces::num_collapsed() {
  return m_num_collapsed;
}

int64_t einsum_ir::backend::IterationSpaces::num_tasks() {
  return m_num_tasks;
}

int64_t const * einsum_ir::backend::IterationSpaces::firsts( int64_t i_task_id ) {
  return m_thread_local_spaces[i_task_id].firsts.data();
}

int64_t const * einsum_ir::backend::IterationSpaces::sizes( int64_t i_task_id ) {
  return m_thread_local_spaces[i_task_id].sizes.data();
}