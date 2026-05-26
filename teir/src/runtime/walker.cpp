#include "teir/exception.h"
#include "teir/ir.h"
#include "teir/primitive.h"
#include "teir/runtime.h"
#include "teir/threading.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace teir {

namespace {

// ============================================================================
// Interned data layout
// ============================================================================

constexpr std::size_t MAX_INTERNED_AXES = 8192;
constexpr std::size_t MAX_ANCESTOR_DEPTH = 128;

using Bases = std::array<void*, MAX_TENSORS_PER_INVOCATION>;

/// Per-depth ancestor iteration index; `evaluate_guard` reads directly by
/// precomputed depth.
using AncestorStack = std::array<int64_t, MAX_ANCESTOR_DEPTH>;

/// Path of (axis_idx, depth) entries from root to the node currently being
/// resolved. Used by `resolve_tree` to fill in `InternedGuardTerm.ancestor_depth`.
using PathStack = std::vector<std::pair<int32_t, int32_t>>;

struct InternedGuardTerm {
  GuardTerm::Kind kind;
  int32_t axis_idx;       ///< Diagnostic-only.
  int32_t ancestor_depth; ///< Index into `AncestorStack`, resolved at prepare.
  int64_t axis_extent;    ///< Cached extent for `LAST` comparison.
};

using InternedGuard = std::vector<InternedGuardTerm>;

struct InternedParallelChain {
  std::vector<int32_t> iter_indices;   ///< Iter-node indices, head first.
  std::vector<int64_t> decode_strides; ///< Per-link decode multiplier.
  int64_t total_extent = 0;            ///< Product of link extents.
};

struct InternedIterationNode {
  int32_t axis_idx = -1;
  int64_t extent = 0;
  Policy policy = Policy::SEQUENTIAL;
  std::vector<int32_t> children_idx;
  InternedGuard guard; ///< Empty when the iteration is unguarded.
  int64_t num_threads = 0;
  std::array<int64_t, MAX_TENSORS_PER_INVOCATION> stride_per_tensor{};
  std::array<int64_t, MAX_TENSORS_PER_INVOCATION> offset_per_tensor{};
  /// Populated for every parallel node. Chain links carry an unused chain;
  /// only the chain head's chain is dispatched (`walk_subtree` enters the
  /// head, never the inner links).
  InternedParallelChain chain;
};

struct InternedInvocationNode {
  CompiledInvocation compiled;
  InternedGuard guard; ///< Empty when the invocation is unguarded.
};

} // namespace

// ============================================================================
// Operation::Impl
// ============================================================================

struct Operation::Impl {
  Teir ir;
  std::string backend;

  std::unordered_map<AxisId, int32_t> axis_index_of;
  std::unordered_map<NodeId, int32_t> node_index_of;

  std::vector<InternedIterationNode> iter_nodes;
  std::vector<InternedInvocationNode> inv_nodes;
  std::vector<int32_t> root_node_indices;

  int32_t num_tensors = 0;
  int32_t num_axes = 0;

  void prepare();
  void execute(const std::vector<void*>& bases);

 private:
  InternedGuard intern_guard_raw(const Guard& guard) const;
  void resolve_guard_terms(InternedGuard& interned, const PathStack& path) const;
  int32_t resolve_tree(int32_t node_idx, PathStack& path, int32_t depth);
  void build_parallel_chains();

  void walk_subtree(int32_t start_node, Bases bases, AncestorStack& stack, int32_t depth) const;
  void invoke(const InternedInvocationNode& inv, const Bases& bases) const;
  void dispatch_parallel(const InternedIterationNode& head,
                         Bases bases,
                         AncestorStack& stack,
                         int32_t depth) const;
  bool evaluate_guard(const InternedGuard& guard, const AncestorStack& stack) const;
};

// ============================================================================
// Prepare
// ============================================================================

void Operation::Impl::prepare() {
  num_tensors = static_cast<int32_t>(ir.tensors.size());
  num_axes = static_cast<int32_t>(ir.axes.size());

  if (static_cast<std::size_t>(num_tensors) > MAX_TENSORS_PER_INVOCATION) {
    throw ValidationException("IR declares " + std::to_string(num_tensors) +
                              " tensors but the walker's TileAddresses array fits at most " +
                              std::to_string(MAX_TENSORS_PER_INVOCATION));
  }
  if (static_cast<std::size_t>(num_axes) > MAX_INTERNED_AXES) {
    throw ValidationException("IR declares " + std::to_string(num_axes) +
                              " axes but the walker supports at most " +
                              std::to_string(MAX_INTERNED_AXES));
  }

  // Index axes by id. The Python validator already verified id uniqueness,
  // schedule acyclicity, and child-list sanity, so we do not re-check here.
  axis_index_of.reserve(static_cast<std::size_t>(num_axes));
  for (int32_t i = 0; i < num_axes; ++i) {
    axis_index_of[ir.axes[static_cast<std::size_t>(i)].id] = i;
  }

  // Resolve each primitive's registered CompileFn.
  std::unordered_map<PrimitiveId, CompileFn> compile_by_primitive;
  compile_by_primitive.reserve(ir.primitives.size());
  for (const auto& prim : ir.primitives) {
    if (!has_primitive(backend, prim.operation)) {
      throw LoweringException("no " + backend + " lowering for operation " + prim.operation);
    }
    compile_by_primitive[prim.id] = lookup_primitive(backend, prim.operation);
  }

  // Assign node indices: iteration nodes first (`[0, iter_nodes.size())`),
  // then invocation nodes. The range determines node kind at walk time,
  // avoiding a side-table.
  iter_nodes.clear();
  iter_nodes.resize(ir.schedule.iterations.size());
  inv_nodes.clear();
  inv_nodes.resize(ir.schedule.invocations.size());
  const std::size_t total_nodes = iter_nodes.size() + inv_nodes.size();
  node_index_of.reserve(total_nodes);

  for (std::size_t i = 0; i < ir.schedule.iterations.size(); ++i) {
    node_index_of[ir.schedule.iterations[i].id] = static_cast<int32_t>(i);
  }
  for (std::size_t i = 0; i < ir.schedule.invocations.size(); ++i) {
    node_index_of[ir.schedule.invocations[i].id] = static_cast<int32_t>(iter_nodes.size() + i);
  }

  // Populate iteration nodes: axis, extent, per-tensor stride/offset rows,
  // children, policy, num_threads, raw guards. Ancestor depths on the
  // guard terms are filled in by `resolve_tree` below.
  for (std::size_t i = 0; i < ir.schedule.iterations.size(); ++i) {
    const IterationNode& src = ir.schedule.iterations[i];
    InternedIterationNode& dst = iter_nodes[i];
    {
      auto it = axis_index_of.find(src.axis);
      if (it == axis_index_of.end()) {
        throw ValidationException("iteration node '" + src.id +
                                  "' references unknown axis '" + src.axis + "'");
      }
      dst.axis_idx = it->second;
    }
    const Axis& axis = ir.axes[static_cast<std::size_t>(dst.axis_idx)];
    dst.extent = axis.extent;
    for (int32_t t = 0; t < num_tensors; ++t) {
      const std::size_t ti = static_cast<std::size_t>(t);
      dst.stride_per_tensor[ti] =
          (ti < axis.strides_by_tensor.size()) ? axis.strides_by_tensor[ti] : 0;
      dst.offset_per_tensor[ti] =
          (ti < axis.offsets_by_tensor.size()) ? axis.offsets_by_tensor[ti] : 0;
    }
    dst.policy = src.policy;
    dst.num_threads = src.num_threads;
    dst.children_idx.reserve(src.children.size());
    for (const NodeId& cid : src.children) {
      auto it = node_index_of.find(cid);
      if (it == node_index_of.end()) {
        throw ValidationException("iteration node '" + src.id +
                                  "' references unknown child node '" + cid + "'");
      }
      dst.children_idx.push_back(it->second);
    }
    if (src.guard.has_value()) {
      dst.guard = intern_guard_raw(*src.guard);
    }
  }

  // Populate invocation nodes: compile each invocation's primitive once
  // at this point so dispatch-failure errors surface here, not at execute.
  std::unordered_map<PrimitiveId, const Primitive*> primitive_by_id;
  primitive_by_id.reserve(ir.primitives.size());
  for (const auto& prim : ir.primitives) {
    primitive_by_id[prim.id] = &prim;
  }
  for (std::size_t i = 0; i < ir.schedule.invocations.size(); ++i) {
    const InvocationNode& src = ir.schedule.invocations[i];
    InternedInvocationNode& dst = inv_nodes[i];
    auto pit = primitive_by_id.find(src.primitive);
    if (pit == primitive_by_id.end()) {
      throw ValidationException("invocation '" + src.id +
                                "' references unknown primitive '" + src.primitive + "'");
    }
    const Primitive& prim = *pit->second;
    auto cit = compile_by_primitive.find(src.primitive);
    if (cit == compile_by_primitive.end()) {
      throw ValidationException("invocation '" + src.id +
                                "' has no compiled function for primitive '" +
                                src.primitive + "'");
    }
    CompileFn fn = cit->second;
    dst.compiled = fn(prim, ir);
    if (src.guard.has_value()) {
      dst.guard = intern_guard_raw(*src.guard);
    }
  }

  root_node_indices.clear();
  root_node_indices.reserve(ir.schedule.roots.size());
  for (const NodeId& rid : ir.schedule.roots) {
    auto it = node_index_of.find(rid);
    if (it == node_index_of.end()) {
      throw ValidationException("schedule root references unknown node '" + rid + "'");
    }
    root_node_indices.push_back(it->second);
  }

  // Walk the schedule tree from each root, resolving each guard term's
  // ancestor depth against the current path. Track the maximum depth at
  // which any iteration node writes the ancestor stack, and verify it
  // fits MAX_ANCESTOR_DEPTH so the walker never has to re-check at execute.
  PathStack path;
  int32_t max_iter_depth = -1;
  for (int32_t root_idx : root_node_indices) {
    max_iter_depth = std::max(max_iter_depth, resolve_tree(root_idx, path, 0));
  }
  if (max_iter_depth >= static_cast<int32_t>(MAX_ANCESTOR_DEPTH)) {
    throw ValidationException("schedule iteration depth " + std::to_string(max_iter_depth + 1) +
                              " exceeds walker maximum of " + std::to_string(MAX_ANCESTOR_DEPTH));
  }

  // Precompute the parallel chain rooted at every parallel iteration node.
  build_parallel_chains();
}

InternedGuard Operation::Impl::intern_guard_raw(const Guard& guard) const {
  InternedGuard out;
  out.reserve(guard.size());
  for (const GuardTerm& term : guard) {
    InternedGuardTerm t;
    t.kind = term.kind;
    auto it = axis_index_of.find(term.axis);
    if (it == axis_index_of.end()) {
      throw ValidationException("guard references unknown axis '" + term.axis + "'");
    }
    t.axis_idx = it->second;
    t.ancestor_depth = -1;
    t.axis_extent = ir.axes[static_cast<std::size_t>(t.axis_idx)].extent;
    out.push_back(t);
  }
  return out;
}

void Operation::Impl::resolve_guard_terms(InternedGuard& interned, const PathStack& path) const {
  for (InternedGuardTerm& term : interned) {
    // Walk the path innermost-first; the first matching ancestor wins when
    // multiple iteration nodes share an axis (the innermost is the binding
    // one for guard semantics).
    auto it = std::find_if(path.rbegin(), path.rend(), [&](const auto& entry) {
      return entry.first == term.axis_idx;
    });
    if (it == path.rend()) {
      throw ValidationException("guard references non-ancestor axis " +
                                ir.axes[static_cast<std::size_t>(term.axis_idx)].id);
    }
    term.ancestor_depth = it->second;
  }
}

int32_t Operation::Impl::resolve_tree(int32_t node_idx, PathStack& path, int32_t depth) {
  if (depth >= static_cast<int32_t>(MAX_ANCESTOR_DEPTH)) {
    throw ValidationException("schedule iteration depth " + std::to_string(depth + 1) +
                              " exceeds walker maximum of " +
                              std::to_string(MAX_ANCESTOR_DEPTH));
  }
  if (static_cast<std::size_t>(node_idx) < iter_nodes.size()) {
    InternedIterationNode& it = iter_nodes[static_cast<std::size_t>(node_idx)];
    if (!it.guard.empty()) {
      resolve_guard_terms(it.guard, path);
    }
    int32_t max_depth = depth;
    path.emplace_back(it.axis_idx, depth);
    for (int32_t cidx : it.children_idx) {
      max_depth = std::max(max_depth, resolve_tree(cidx, path, depth + 1));
    }
    path.pop_back();
    return max_depth;
  }
  InternedInvocationNode& inv = inv_nodes[static_cast<std::size_t>(node_idx) - iter_nodes.size()];
  if (!inv.guard.empty()) {
    resolve_guard_terms(inv.guard, path);
  }
  // Invocations do not write the ancestor stack; they do not contribute
  // to the iteration-depth limit.
  return -1;
}

void Operation::Impl::build_parallel_chains() {
  for (std::size_t i = 0; i < iter_nodes.size(); ++i) {
    InternedIterationNode& head = iter_nodes[i];
    if (head.policy != Policy::PARALLEL) {
      continue;
    }
    head.chain.iter_indices.push_back(static_cast<int32_t>(i));
    int32_t cur = static_cast<int32_t>(i);
    while (true) {
      const InternedIterationNode& cur_node = iter_nodes[static_cast<std::size_t>(cur)];
      if (cur_node.children_idx.size() != 1) {
        break;
      }
      const int32_t child_idx = cur_node.children_idx[0];
      if (static_cast<std::size_t>(child_idx) >= iter_nodes.size()) {
        break;
      }
      const InternedIterationNode& child = iter_nodes[static_cast<std::size_t>(child_idx)];
      if (child.policy != Policy::PARALLEL || !child.guard.empty()) {
        break;
      }
      head.chain.iter_indices.push_back(child_idx);
      cur = child_idx;
    }
    const std::size_t chain_len = head.chain.iter_indices.size();
    int64_t total = 1;
    for (int32_t idx : head.chain.iter_indices) {
      total *= iter_nodes[static_cast<std::size_t>(idx)].extent;
    }
    head.chain.total_extent = total;
    head.chain.decode_strides.assign(chain_len, 1);
    for (std::size_t k = chain_len - 1; k > 0; --k) {
      head.chain.decode_strides[k - 1] =
          head.chain.decode_strides[k] *
          iter_nodes[static_cast<std::size_t>(head.chain.iter_indices[k])].extent;
    }
  }
}

// ============================================================================
// Execute
// ============================================================================

bool Operation::Impl::evaluate_guard(const InternedGuard& guard, const AncestorStack& stack) const {
  for (const InternedGuardTerm& term : guard) {
    const int64_t cur_idx = stack[static_cast<std::size_t>(term.ancestor_depth)];
    if (term.kind == GuardTerm::Kind::FIRST) {
      if (cur_idx != 0) {
        return false;
      }
    } else if (cur_idx != term.axis_extent - 1) {
      return false;
    }
  }
  return true;
}

void Operation::Impl::invoke(const InternedInvocationNode& inv, const Bases& bases) const {
  TileAddresses tiles;
  tiles.num_tensors = num_tensors;
  for (int32_t t = 0; t < num_tensors; ++t) {
    tiles.addrs[static_cast<std::size_t>(t)] = bases[static_cast<std::size_t>(t)];
  }
  inv.compiled.kernel(tiles, inv.compiled.state.get());
}

void Operation::Impl::walk_subtree(int32_t node_idx,
                                   Bases bases,
                                   AncestorStack& stack,
                                   int32_t depth) const {
  if (static_cast<std::size_t>(node_idx) < iter_nodes.size()) {
    const InternedIterationNode& it = iter_nodes[static_cast<std::size_t>(node_idx)];
    if (!it.guard.empty() && !evaluate_guard(it.guard, stack)) {
      return;
    }
    if (it.policy == Policy::PARALLEL) {
      dispatch_parallel(it, bases, stack, depth);
      return;
    }
    if (it.extent <= 0) {
      return;
    }
    // Depth fits MAX_ANCESTOR_DEPTH by construction; verified at prepare.
    // Apply offset once, then increment by stride per iteration. `bases`
    // is our local copy by value; the caller's view is untouched.
    for (int32_t t = 0; t < num_tensors; ++t) {
      bases[static_cast<std::size_t>(t)] = static_cast<char*>(bases[static_cast<std::size_t>(t)]) +
                                           it.offset_per_tensor[static_cast<std::size_t>(t)];
    }
    for (int64_t i = 0; i < it.extent; ++i) {
      stack[static_cast<std::size_t>(depth)] = i;
      for (int32_t cidx : it.children_idx) {
        walk_subtree(cidx, bases, stack, depth + 1);
      }
      for (int32_t t = 0; t < num_tensors; ++t) {
        bases[static_cast<std::size_t>(t)] =
            static_cast<char*>(bases[static_cast<std::size_t>(t)]) +
            it.stride_per_tensor[static_cast<std::size_t>(t)];
      }
    }
    return;
  }

  const InternedInvocationNode& inv =
      inv_nodes[static_cast<std::size_t>(node_idx) - iter_nodes.size()];
  if (!inv.guard.empty() && !evaluate_guard(inv.guard, stack)) {
    return;
  }
  invoke(inv, bases);
}

void Operation::Impl::dispatch_parallel(const InternedIterationNode& head,
                                        Bases bases,
                                        AncestorStack& stack,
                                        int32_t depth) const {
  const InternedParallelChain& chain = head.chain;
  const std::size_t chain_len = chain.iter_indices.size();
  if (chain.total_extent <= 0) {
    return;
  }
  // depth + chain_len fits MAX_ANCESTOR_DEPTH by construction; verified at prepare.

  const int32_t leaf_idx = chain.iter_indices.back();
  const std::vector<int32_t>& leaf_children =
      iter_nodes[static_cast<std::size_t>(leaf_idx)].children_idx;
  const int32_t local_depth = depth + static_cast<int32_t>(chain_len);

  auto walk_worker = [&](int64_t i, Bases worker_bases, AncestorStack& worker_stack) {
    for (std::size_t k = 0; k < chain_len; ++k) {
      const InternedIterationNode& link =
          iter_nodes[static_cast<std::size_t>(chain.iter_indices[k])];
      const int64_t i_k = (i / chain.decode_strides[k]) % link.extent;
      for (int32_t t = 0; t < num_tensors; ++t) {
        const int64_t step = link.stride_per_tensor[static_cast<std::size_t>(t)] * i_k +
                             link.offset_per_tensor[static_cast<std::size_t>(t)];
        worker_bases[static_cast<std::size_t>(t)] =
            static_cast<char*>(worker_bases[static_cast<std::size_t>(t)]) + step;
      }
      worker_stack[static_cast<std::size_t>(depth) + k] = i_k;
    }
    for (int32_t cidx : leaf_children) {
      walk_subtree(cidx, worker_bases, worker_stack, local_depth);
    }
  };

  // Single-thread fast path: skip the threading layer entirely. `stack` is
  // owned by this call frame and safely mutable across iterations because
  // the workers run sequentially and write disjoint depth slices. The
  // bypass triggers only on an explicit ``num_threads == 1`` request;
  // ``num_threads == 0`` means "backend default" and falls through to
  // ``execute_threaded``, which resolves zero to all available
  // workers.
  if (head.num_threads == 1) {
    for (int64_t i = 0; i < chain.total_extent; ++i) {
      walk_worker(i, bases, stack);
    }
    return;
  }

  std::atomic<bool> failed{false};
  std::exception_ptr captured;
  std::mutex captured_mutex;
  threading::execute_threaded(chain.total_extent, head.num_threads, [&](int64_t i) {
    if (failed.load(std::memory_order_acquire)) {
      return;
    }
    try {
      AncestorStack local_stack = stack;
      walk_worker(i, bases, local_stack);
    } catch (...) {
      std::scoped_lock guard{captured_mutex};
      if (!captured) {
        captured = std::current_exception();
        failed.store(true, std::memory_order_release);
      }
    }
  });
  if (captured) {
    std::rethrow_exception(captured);
  }
}

void Operation::Impl::execute(const std::vector<void*>& bases) {
  if (static_cast<int32_t>(bases.size()) != num_tensors) {
    throw RuntimeException("tensor count mismatch in execute()");
  }
  Bases work{};
  for (int32_t t = 0; t < num_tensors; ++t) {
    work[static_cast<std::size_t>(t)] = bases[static_cast<std::size_t>(t)];
  }
  AncestorStack stack{};
  for (int32_t rid : root_node_indices) {
    walk_subtree(rid, work, stack, 0);
  }
}

// ============================================================================
// Public API
// ============================================================================

Operation::Operation(Teir ir, std::string backend) : impl_(std::make_unique<Impl>()) {
  impl_->ir = std::move(ir);
  impl_->backend = std::move(backend);
  impl_->prepare();
}

Operation::~Operation() = default;
Operation::Operation(Operation&&) noexcept = default;
Operation& Operation::operator=(Operation&&) noexcept = default;

const Teir& Operation::ir() const noexcept {
  return impl_->ir;
}
const std::string& Operation::backend() const noexcept {
  return impl_->backend;
}

void Operation::execute(const std::vector<void*>& bases) {
  impl_->execute(bases);
}

std::unique_ptr<Operation> compile(Teir ir, std::string_view backend) {
  return std::make_unique<Operation>(std::move(ir), std::string(backend));
}

} // namespace teir
