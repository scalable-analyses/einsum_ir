/// @file runtime.h
/// @brief Compile a `Teir` for a chosen backend and execute it.
#pragma once

#include "teir/ir.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace teir {

/// A compiled, executable operation.
class Operation {
 public:
  Operation(Teir ir, std::string backend);
  Operation(const Operation&) = delete;
  Operation& operator=(const Operation&) = delete;
  Operation(Operation&&) noexcept;
  Operation& operator=(Operation&&) noexcept;
  ~Operation();

  /// Walk the schedule using the tensor base pointers in `bases`. The number
  /// of pointers must match the number of tensors in the IR; order matches
  /// `ir.tensors`.
  void execute(const std::vector<void*>& bases);

  /// Borrowed reference to the compiled IR. Undefined behavior if the
  /// `Operation` has been moved from; call sites that hold an `Operation`
  /// by reference should assume the move-from precondition is satisfied.
  [[nodiscard]] const Teir& ir() const noexcept;
  [[nodiscard]] const std::string& backend() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Compile a `Teir` for the named backend. Currently recognized backends:
/// "tpp", "blas". Returns a heap-allocated `Operation` to keep the move-only
/// guarantee stable across the pybind boundary.
[[nodiscard]] std::unique_ptr<Operation> compile(Teir ir, std::string_view backend);

} // namespace teir
