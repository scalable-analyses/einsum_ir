/// @file hash.h
/// @brief Shared hash combiner used by the runtime caches.
#pragma once

#include <cstddef>
#include <cstdint>

namespace teir::internal {

constexpr std::size_t hash_combine(std::size_t seed, std::size_t value) noexcept {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

} // namespace teir::internal
