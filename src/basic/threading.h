#ifndef EINSUM_IR_BASIC_THREADING
#define EINSUM_IR_BASIC_THREADING

#include <cstdint>

#if defined(EINSUM_IR_USE_DISPATCH)
  #include <dispatch/dispatch.h>
  #include <sys/sysctl.h>
#elif defined(EINSUM_IR_USE_OPENMP)
  #include <omp.h>
#endif

namespace einsum_ir {
  namespace basic {
    
    /**
     * @brief Get available hardware thread count
     *
     * On Apple Silicon, returns P-core (Performance cores) count only.
     *
     * @return Number of performance cores (Apple Silicon) or logical CPUs (Intel) (always >= 1)
     **/
    inline int64_t get_num_threads_available() {
#if defined(EINSUM_IR_USE_DISPATCH)
      // macOS/iOS: Query P-core count on Apple Silicon
      // hw.perflevel0.physicalcpu = Physical P-cores
      int l_count = 0;
      size_t l_size = sizeof(l_count);
      
      // Try P-cores first (Apple Silicon)
      if( sysctlbyname("hw.perflevel0.physicalcpu", &l_count, &l_size, nullptr, 0) == 0 && l_count > 0 ) {
        return static_cast<int64_t>(l_count);
      }

      // Fallback to all logical CPUs (Intel Macs or older macOS)
      if( sysctlbyname("hw.logicalcpu", &l_count, &l_size, nullptr, 0) == 0 && l_count > 0 ) {
        return static_cast<int64_t>(l_count);
      }

      return 1;  // Final fallback

#elif defined(EINSUM_IR_USE_OPENMP)
      // OpenMP: Use standard API (respects OMP_NUM_THREADS)
      return static_cast<int64_t>(omp_get_max_threads());

#else
      // Sequential: Single thread
      return 1;
#endif
    }

    /**
     * @brief Execute work function using threaded workers
     *
     * @tparam WorkFunc Callable with signature void(int64_t thread_id)
     * @param i_num_threads Number of threads to spawn
     * @param i_work Work function, called once per thread with thread ID in [0, i_num_threads)
     **/
    template<typename WorkFunc>
    inline void execute_threaded( int64_t   i_num_threads,
                                  WorkFunc  i_work ) {
      // Sequential fallback for single-threaded case
      if( i_num_threads <= 1 ) {
        i_work( 0 );
        return;
      }

#if defined(EINSUM_IR_USE_DISPATCH)
      // Apple Dispatch implementation
      dispatch_queue_attr_t l_attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_CONCURRENT,
        QOS_CLASS_USER_INTERACTIVE,
        0 );
      
      dispatch_queue_t l_queue = dispatch_queue_create_with_target(
        "einsum_ir.parallel",
        l_attr,
        dispatch_get_global_queue( QOS_CLASS_USER_INTERACTIVE, 0 ) );

      // dispatch_apply guarantees iteration index == thread ID
      dispatch_apply( static_cast<size_t>(i_num_threads), 
                      l_queue, 
                      ^(size_t l_id) {
                        i_work( static_cast<int64_t>(l_id) );
                      });
      
      dispatch_release( l_queue );
      
#elif defined(EINSUM_IR_USE_OPENMP)
      // OpenMP implementation
#pragma omp parallel for num_threads(i_num_threads)
      for( int64_t l_thread_id = 0; l_thread_id < i_num_threads; l_thread_id++ ) {
        i_work( l_thread_id );
      }
      
#else
      // Sequential fallback
      for( int64_t l_thread_id = 0; l_thread_id < i_num_threads; l_thread_id++ ) {
        i_work( l_thread_id );
      }
#endif
    }
    
  }
}

#endif
