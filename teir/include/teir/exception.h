/// @file exception.h
/// @brief Typed exception hierarchy thrown by the `teir::` runtime.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace teir {

/// Base class for every exception thrown by the runtime.
class Exception : public std::runtime_error {
 public:
  explicit Exception(std::string message) :
      std::runtime_error(message), message_(std::move(message)) {}

  [[nodiscard]] const std::string& message() const noexcept { return message_; }

 private:
  std::string message_;
};

/// Validation failure: a malformed IR was presented to the runtime.
class ValidationException : public Exception {
 public:
  using Exception::Exception;
};

/// Lowering failure: a primitive cannot be lowered for the chosen backend.
class LoweringException : public Exception {
 public:
  using Exception::Exception;
};

/// Runtime failure during execution (kernel dispatch, threading, etc.).
class RuntimeException : public Exception {
 public:
  using Exception::Exception;
};

} // namespace teir
