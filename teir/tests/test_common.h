/// Shared utilities for the C++ test suite.

#pragma once

#include <cstdint>
#include <type_traits>

namespace teir::tests {

inline constexpr int64_t SEED = 0xC0FFEE;

template <typename T>
constexpr double dispatch_atol(int64_t k_total) {
  // Conservative bound on the worst-case error after k_total fused
  // multiply-adds against unit-bounded inputs.
  constexpr double eps = std::is_same_v<T, float> ? 1.0e-6 : 1.0e-14;
  return 8.0 * eps * static_cast<double>(k_total);
}

} // namespace teir::tests
