/// @file primitive.h
/// @brief Backend-pluggable primitive lowering registry.
#pragma once

#include "teir/ir.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace teir {

/// Maximum number of tensors that may participate in a single primitive
/// invocation; raise when additional primitives need more.
inline constexpr std::size_t MAX_TENSORS_PER_INVOCATION = 8;

/// Per-tensor tile addresses passed to a lowered primitive.
struct TileAddresses {
  std::array<void*, MAX_TENSORS_PER_INVOCATION> addrs{};
  int32_t num_tensors = 0;
};

/// Per-invocation kernel function pointer signature.
///
/// `state` is the backend-owned per-invocation payload built at compile
/// time (typically holds the JIT'd kernel function and the tile-index
/// mapping). The walker calls this directly on the hot path.
using KernelFn = void (*)(const TileAddresses& tiles, const void* state);

/// Result of compiling one invocation: a ready-to-call kernel plus the
/// backend-owned state it consumes. The state pointer outlives the call
/// for the operation's lifetime via the unique_ptr's custom deleter.
struct CompiledInvocation {
  KernelFn kernel = nullptr;
  std::unique_ptr<void, void (*)(void*)> state{nullptr, [](void*) {}};
};

/// Backend lowering signature. Called once per invocation node during
/// `Operation::prepare()`; returns the kernel + state pair the walker
/// will invoke for that node. Throws `LoweringException` if the
/// primitive's shape cannot be JIT'd.
using CompileFn = CompiledInvocation (*)(const Primitive& prim, const Teir& teir);

/// Register a backend lowering for an operation.
void register_primitive(std::string_view backend, std::string_view operation, CompileFn fn);

/// Look up a lowering for an operation; throws `LoweringException` if none
/// is registered.
[[nodiscard]] CompileFn lookup_primitive(std::string_view backend, std::string_view operation);

/// Returns true when a lowering is registered.
[[nodiscard]] bool has_primitive(std::string_view backend, std::string_view operation);

namespace tpp {
/// Register the TPP-backend primitive lowerings. Called once at module
/// initialization. No-op when libxsmm support is not compiled in.
void register_tpp_primitives();
} // namespace tpp

namespace blas {
/// Register the BLAS-backend primitive lowerings. Called once at module
/// initialization. No-op when BLAS support is not compiled in.
void register_blas_primitives();
} // namespace blas

} // namespace teir
