/// @file threading.h
/// @brief Platform threading abstraction.
#pragma once

#include "teir/config.h"

#include <cstdint>
#include <functional>

namespace teir::threading {

/// Worker threads available to the runtime.
///
/// OpenMP returns `omp_get_max_threads()`; Dispatch returns
/// `std::thread::hardware_concurrency()`; the sequential backend returns 1.
[[nodiscard]] int64_t num_threads_available();

/// Per-iteration callback.
using WorkFunc = std::function<void(int64_t /*iteration*/)>;

/// Dispatch `num_iterations` work items across at most `num_threads` workers.
/// Passing `num_threads == 0` lets the backend choose; values exceeding
/// `num_threads_available()` are clamped.
void execute_threaded(int64_t num_iterations, int64_t num_threads, const WorkFunc& work);

} // namespace teir::threading
