/// @file ir.h
/// @brief Runtime-side mirror of the Python TEIR data model.
#pragma once

#include "teir/config.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace teir {

/// Identifier of a tensor (e.g. "in0", "in1", "out").
using TensorId = std::string;
/// Identifier of a TEIR axis.
using AxisId = std::string;
/// Identifier of a schedule node (iteration or invocation).
using NodeId = std::string;
/// Identifier of a primitive specification.
using PrimitiveId = std::string;
/// Role label inside a primitive ("M", "N", "K", "C").
using RoleId = std::string;

/// Element data type carried by a tensor.
struct DataType {
  std::string name;
  int64_t bits = 0;

  [[nodiscard]] int64_t bytes() const noexcept { return (bits + 7) / 8; }
};

struct Tensor {
  TensorId id;
  DataType dtype;
};

/// A TEIR axis with per-tensor byte strides and offsets.
///
/// Strides are added to a tensor's tile address each time the schedule
/// iterates this axis from an ancestor iteration node.
///
/// Offsets are *scoped*: an axis's offset for tensor `i` is added to that
/// tensor's tile address only when the axis is iterated by an ancestor in
/// the current schedule path. Axes that appear only in primitive role
/// lists (or that exist in `Teir::axes` but are unreferenced by the
/// schedule) contribute zero offset to every tile address. Python-side
/// validation rejects any axis whose offsets are non-zero on any tensor
/// when that axis appears in any primitive role list.
struct Axis {
  AxisId id;
  int64_t extent = 0;
  /// `strides_by_tensor[i]` is the byte stride on tensor index `i`.
  std::vector<int64_t> strides_by_tensor;
  /// `offsets_by_tensor[i]` is the byte offset on tensor index `i`.
  std::vector<int64_t> offsets_by_tensor;
};

/// A primitive specification.
struct Primitive {
  PrimitiveId id;
  std::string operation;
  /// Role label → ordered list of axis ids consumed at that role.
  std::vector<std::pair<RoleId, std::vector<AxisId>>> axes_by_role;
  /// Operation-specific metadata (`data_type`, …).
  std::vector<std::pair<std::string, std::string>> metadata;
};

/// Guard term: `first(node)` or `last(node)`.
struct GuardTerm {
  enum class Kind : uint8_t { FIRST = 0, LAST = 1 };
  Kind kind;
  NodeId node;
};

using Guard = std::vector<GuardTerm>;

/// Iteration policy of an iteration node.
enum class Policy : uint8_t {
  SEQUENTIAL = 0,
  PARALLEL = 1,
};

struct IterationNode {
  NodeId id;
  AxisId axis;
  Policy policy = Policy::SEQUENTIAL;
  std::vector<NodeId> children;
  std::optional<Guard> guard;

  // ---- Metadata ----
  // Fields below originate from Python `IterationNode.metadata` and are
  // promoted to typed fields at the marshaling boundary (see
  // `etops/lowering/_marshal.py`). Keep typed when the runtime depends on
  // the value; new advisory hints that the runtime ignores belong in
  // Python-side metadata only, not here.
  int64_t num_threads = 0; ///< 0 → backend default.
};

struct InvocationNode {
  NodeId id;
  PrimitiveId primitive;
  std::optional<Guard> guard;
};

struct Schedule {
  std::vector<NodeId> roots;
  std::vector<IterationNode> iterations;
  std::vector<InvocationNode> invocations;
};

/// Full TEIR configuration.
struct Teir {
  std::vector<Tensor> tensors;
  std::vector<Axis> axes;
  std::vector<Primitive> primitives;
  Schedule schedule;
};

} // namespace teir
