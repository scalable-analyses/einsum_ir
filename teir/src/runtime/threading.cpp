#include "teir/threading.h"

#include "teir/config.h"

#include <algorithm>
#include <cstdint>
#include <thread>

#if TEIR_THREADING_BACKEND == TEIR_THREADING_BACKEND_DISPATCH
#  include <dispatch/dispatch.h>
#endif

#if TEIR_THREADING_BACKEND == TEIR_THREADING_BACKEND_OPENMP
#  include <omp.h>
#endif

namespace teir::threading {

namespace {

int64_t resolve_num_threads(int64_t requested) {
  const int64_t available = num_threads_available();
  if (requested <= 0) {
    return available;
  }
  return std::min(requested, available);
}

void run_sequential(int64_t num_iterations, const WorkFunc& work) {
  for (int64_t i = 0; i < num_iterations; ++i) {
    work(i);
  }
}

} // namespace

int64_t num_threads_available() {
#if TEIR_THREADING_BACKEND == TEIR_THREADING_BACKEND_OPENMP
  return static_cast<int64_t>(omp_get_max_threads());
#elif TEIR_THREADING_BACKEND == TEIR_THREADING_BACKEND_DISPATCH
  const auto hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int64_t>(hw) : 1;
#else
  return 1;
#endif
}

void execute_threaded(int64_t num_iterations, int64_t num_threads, const WorkFunc& work) {
  if (num_iterations <= 0) {
    return;
  }
  const int64_t threads = resolve_num_threads(num_threads);
  if (threads <= 1) {
    run_sequential(num_iterations, work);
    return;
  }

#if TEIR_THREADING_BACKEND == TEIR_THREADING_BACKEND_OPENMP
#  pragma omp parallel for num_threads(static_cast<int>(threads)) schedule(static)
  for (int64_t i = 0; i < num_iterations; ++i) {
    work(i);
  }
#elif TEIR_THREADING_BACKEND == TEIR_THREADING_BACKEND_DISPATCH
  dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  __block const WorkFunc* fn = &work;
  dispatch_apply(static_cast<std::size_t>(num_iterations), queue, ^(std::size_t i) {
    (*fn)(static_cast<int64_t>(i));
  });
  (void)threads;
#else
  run_sequential(num_iterations, work);
#endif
}

} // namespace teir::threading
