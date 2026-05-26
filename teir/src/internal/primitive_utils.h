/// @file primitive_utils.h
/// @brief Backend-agnostic helpers shared by TPP and BLAS primitive lowerings.
#pragma once

#include "teir/exception.h"
#include "teir/ir.h"

#include <cstdint>
#include <string>
#include <vector>

namespace teir::internal {

/// Return the ``data_type`` metadata value. Throws if the key is absent.
inline std::string data_type_of(const Primitive& prim) {
  for (const auto& [k, v] : prim.metadata) {
    if (k == "data_type") {
      return v;
    }
  }
  throw LoweringException("primitive " + prim.id + " is missing required `data_type` metadata");
}

/// Byte width for the dispatchable dtypes. Throws on anything else.
inline int64_t dtype_bytes(const std::string& backend, const std::string& name) {
  if (name == "f32") {
    return 4;
  }
  if (name == "f64") {
    return 8;
  }
  throw LoweringException(backend + ": unsupported data type " + name);
}

/// Locate an `Axis` record by id. Throws when the id is unknown.
inline const Axis& axis_by_id(const std::string& backend, const Teir& teir, const AxisId& id) {
  for (const auto& a : teir.axes) {
    if (a.id == id) {
      return a;
    }
  }
  throw LoweringException(backend + ": unknown axis " + id);
}

/// Byte stride of ``a`` on ``tensor_idx`` (zero if out-of-range).
inline int64_t stride_of(const Axis& a, int32_t tensor_idx) noexcept {
  if (tensor_idx < 0 || static_cast<std::size_t>(tensor_idx) >= a.strides_by_tensor.size()) {
    return 0;
  }
  return a.strides_by_tensor[static_cast<std::size_t>(tensor_idx)];
}

/// Ordered axis list for a named role on ``prim``. Returns an empty list
/// when the role is unknown to that primitive.
inline const std::vector<AxisId>& role(const Primitive& prim, const std::string& name) {
  static const std::vector<AxisId> empty;
  for (const auto& [r, axes] : prim.axes_by_role) {
    if (r == name) {
      return axes;
    }
  }
  return empty;
}

/// Per-operand axis pair fed to `resolve_operand`. ``primary`` is the
/// axis libxsmm / cblas wants unit-stride; ``secondary`` is the axis
/// that becomes the leading dimension. If TEIR has ``secondary`` unit-
/// stride instead, the operand is transposed and the axes swap roles.
struct OperandStrides {
  int64_t primary;   ///< byte stride of the axis libxsmm/cblas wants unit-stride
  int64_t secondary; ///< byte stride of the axis that becomes the leading dim
};

/// Classify one GEMM operand. Given two byte strides for its two role
/// axes, identify which is the unit-stride axis and report:
///   - ``ld_bytes``: the *other* axis's byte stride (the leading dim).
///   - ``transposed``: ``true`` iff the secondary axis was unit-stride
///     (i.e., the operand needs a TRANS flag at dispatch).
/// Returns ``false`` when neither stride equals ``bytes``.
[[nodiscard]] inline bool
resolve_operand(OperandStrides s, int64_t bytes, int64_t& ld_bytes, bool& transposed) noexcept {
  if (s.primary == bytes) {
    ld_bytes = s.secondary;
    transposed = false;
    return true;
  }
  if (s.secondary == bytes) {
    ld_bytes = s.primary;
    transposed = true;
    return true;
  }
  return false;
}

} // namespace teir::internal
