/// TPP backend primitive lowerings.
///
/// All TEIR primitives — Zero, Copy, ReLU, Contraction — dispatch exclusively
/// through libxsmm. The optimization pipeline is responsible for arranging
/// primitive role cardinalities and stride patterns into a
/// libxsmm-dispatchable shape. When libxsmm declines a shape the lowering
/// raises `LoweringException`.
///
/// Each `CompileFn` JITs its kernel once at prepare time via
/// `libxsmm_dispatch_meltw_unary` / `libxsmm_dispatch_gemm` /
/// `libxsmm_dispatch_brgemm`, stashes the resulting function pointer on a
/// `CompiledInvocation` state struct, and returns a trampoline kernel that
/// calls it. libxsmm itself dedupes by descriptor in a thread-local L1 +
/// lock-free global registry, so two invocations with identical shapes share
/// one JIT'd function.

#include "internal/primitive_utils.h"
#include "teir/config.h"
#include "teir/exception.h"
#include "teir/ir.h"
#include "teir/primitive.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#if ETOPS_LIBXSMM_AVAILABLE
#  include <libxsmm.h>
#endif

namespace teir::tpp {

namespace {

constexpr const char* BACKEND = "tpp";

inline const Axis& axis_by_id(const Teir& teir, const AxisId& id) {
  return internal::axis_by_id(BACKEND, teir, id);
}

inline int64_t dtype_bytes(const std::string& name) {
  return internal::dtype_bytes(BACKEND, name);
}

using internal::data_type_of;
using internal::OperandStrides;
using internal::resolve_operand;
using internal::role;
using internal::stride_of;

#if ETOPS_LIBXSMM_AVAILABLE
libxsmm_datatype to_libxsmm_dtype(const std::string& name) {
  if (name == "f32") {
    return LIBXSMM_DATATYPE_F32;
  }
  if (name == "f64") {
    return LIBXSMM_DATATYPE_F64;
  }
  return LIBXSMM_DATATYPE_UNSUPPORTED;
}
#endif

#if ETOPS_LIBXSMM_AVAILABLE

// --------------------------------------------------------------------------
// libxsmm initialization
// --------------------------------------------------------------------------

void ensure_libxsmm_init() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { libxsmm_init(); });
}

bool fits_in_blasint(int64_t value) {
  return value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<libxsmm_blasint>::max());
}

// --------------------------------------------------------------------------
// Tile layout classification
// --------------------------------------------------------------------------

struct TileLayout {
  // libxsmm meltw_unary speaks column-major (m, n) tiles with unit-stride
  // along m and stride-ld along n. We project the role-axis tile onto that
  // layout by picking the unit-stride axis as m and (optionally) a second
  // axis as n. ``unit_axis`` is the role-axis index whose byte stride is
  // exactly ``bytes``; ``ld_axis`` is the other active axis (or -1 if
  // none).
  int64_t m = 1;
  int64_t n = 1;
  int64_t ld = 1;
  int32_t unit_axis = -1;
  int32_t ld_axis = -1;
};

bool classify_operand_layout(const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& extents,
                             int64_t bytes,
                             TileLayout& layout) {
  std::vector<int32_t> active;
  active.reserve(extents.size());
  for (int32_t i = 0; i < static_cast<int32_t>(extents.size()); ++i) {
    if (extents[i] > 1) {
      active.push_back(i);
    }
  }
  if (active.size() > 2) {
    return false;
  }
  if (active.empty()) {
    layout.m = 1;
    layout.n = 1;
    layout.ld = 1;
    layout.unit_axis = -1;
    layout.ld_axis = -1;
    return true;
  }
  int32_t unit = -1;
  int32_t other = -1;
  for (int32_t idx : active) {
    if (strides[idx] == bytes) {
      unit = idx;
    } else {
      other = idx;
    }
  }
  if (unit < 0) {
    return false;
  }
  layout.unit_axis = unit;
  layout.m = extents[unit];
  if (other < 0) {
    layout.n = 1;
    layout.ld = layout.m;
    layout.ld_axis = -1;
    return true;
  }
  if (strides[other] % bytes != 0) {
    return false;
  }
  layout.n = extents[other];
  layout.ld = strides[other] / bytes;
  layout.ld_axis = other;
  return true;
}

void collect_role_strides(const Primitive& prim,
                          const Teir& teir,
                          const std::string& role_name,
                          int32_t in_tensor_idx,
                          int32_t out_tensor_idx,
                          std::vector<int64_t>& in_strides,
                          std::vector<int64_t>& out_strides,
                          std::vector<int64_t>& extents) {
  const auto& axes = role(prim, role_name);
  for (const auto& a : axes) {
    const Axis& ax = axis_by_id(teir, a);
    in_strides.push_back(stride_of(ax, in_tensor_idx));
    out_strides.push_back(stride_of(ax, out_tensor_idx));
    extents.push_back(ax.extent);
  }
}

// --------------------------------------------------------------------------
// Per-invocation state + kernel trampolines
// --------------------------------------------------------------------------

struct UnaryState {
  libxsmm_meltwfunction_unary fn;
  int32_t in_tensor_idx;
  int32_t out_tensor_idx;
};

void unary_kernel(const TileAddresses& tiles, const void* state) {
  const auto* s = static_cast<const UnaryState*>(state);
  libxsmm_meltw_unary_param param{};
  param.in.primary = tiles.addrs[static_cast<std::size_t>(s->in_tensor_idx)];
  param.out.primary = tiles.addrs[static_cast<std::size_t>(s->out_tensor_idx)];
  s->fn(&param);
}

struct ContractionState {
  libxsmm_gemmfunction fn;
  int32_t a_tensor_idx;
  int32_t b_tensor_idx;
  int32_t c_tensor_idx;
  int64_t br_count; ///< 0 = plain GEMM; > 0 = BRGEMM batch count.
};

void contraction_kernel(const TileAddresses& tiles, const void* state) {
  const auto* s = static_cast<const ContractionState*>(state);
  libxsmm_gemm_param param{};
  param.a.primary = tiles.addrs[static_cast<std::size_t>(s->a_tensor_idx)];
  param.b.primary = tiles.addrs[static_cast<std::size_t>(s->b_tensor_idx)];
  param.c.primary = tiles.addrs[static_cast<std::size_t>(s->c_tensor_idx)];
  unsigned long long batch_count;
  if (s->br_count > 0) {
    batch_count = static_cast<unsigned long long>(s->br_count);
    param.op.tertiary = &batch_count;
  }
  s->fn(&param);
}

template <typename T>
void state_deleter(void* p) {
  delete static_cast<T*>(p);
}

CompiledInvocation compile_unary(libxsmm_meltw_unary_type op,
                                 const std::string& dtype,
                                 const std::string& op_name,
                                 const std::string& prim_id,
                                 int64_t m,
                                 int64_t n,
                                 int64_t ldi,
                                 int64_t ldo,
                                 int32_t in_idx,
                                 int32_t out_idx) {
  ensure_libxsmm_init();
  const libxsmm_datatype dt = to_libxsmm_dtype(dtype);
  libxsmm_meltwfunction_unary fn = nullptr;
  if (dt != LIBXSMM_DATATYPE_UNSUPPORTED && fits_in_blasint(m) && fits_in_blasint(n) &&
      fits_in_blasint(ldi) && fits_in_blasint(ldo)) {
    libxsmm_meltw_unary_shape shape{};
    shape.m = static_cast<libxsmm_blasint>(m);
    shape.n = static_cast<libxsmm_blasint>(n);
    shape.ldi = static_cast<libxsmm_blasint>(ldi);
    shape.ldo = static_cast<libxsmm_blasint>(ldo);
    shape.in0_type = dt;
    shape.out_type = dt;
    shape.comp_type = dt;
    fn = libxsmm_dispatch_meltw_unary(op, shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    // When libxsmm refuses to JIT a (op, dtype) combination it silently
    // substitutes a C reference kernel whose element get/set helpers are
    // float-only; for f64 (and likely other non-f32 dtypes) the helpers
    // no-op and the output buffer is never written. Refuse the kernel so
    // the LoweringException below fires, rather than producing silently
    // wrong results at execute time.
    libxsmm_kernel_info info{};
    if (fn != nullptr && libxsmm_get_kernel_info(reinterpret_cast<const void*>(fn), &info) == 0 &&
        info.is_reference_kernel != 0) {
      fn = nullptr;
    }
  }
  if (fn == nullptr) {
    throw LoweringException("tpp: libxsmm could not dispatch a " + op_name +
                            " kernel for primitive '" + prim_id + "' (dtype=" + dtype +
                            ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ")");
  }
  auto* state = new UnaryState{.fn = fn, .in_tensor_idx = in_idx, .out_tensor_idx = out_idx};
  return CompiledInvocation{
      .kernel = &unary_kernel,
      .state = {state, &state_deleter<UnaryState>},
  };
}

CompiledInvocation compile_zero(const std::string& dtype,
                                int32_t out_idx,
                                const std::vector<int64_t>& out_strides,
                                const std::vector<int64_t>& extents,
                                const std::string& prim_id) {
  const int64_t bytes = dtype_bytes(dtype);
  TileLayout out_layout;
  if (!classify_operand_layout(out_strides, extents, bytes, out_layout)) {
    throw LoweringException("tpp: Zero primitive '" + prim_id +
                            "' requires a tile layout with at most two axes of extent > 1 and "
                            "a unit-stride innermost axis on the output tensor");
  }
  return compile_unary(LIBXSMM_MELTW_TYPE_UNARY_XOR,
                       dtype,
                       "Zero",
                       prim_id,
                       out_layout.m,
                       out_layout.n,
                       out_layout.ld,
                       out_layout.ld,
                       out_idx,
                       out_idx);
}

CompiledInvocation compile_copy(const std::string& dtype,
                                int32_t in_idx,
                                int32_t out_idx,
                                const std::vector<int64_t>& in_strides,
                                const std::vector<int64_t>& out_strides,
                                const std::vector<int64_t>& extents,
                                const std::string& prim_id) {
  const int64_t bytes = dtype_bytes(dtype);
  TileLayout in_layout;
  TileLayout out_layout;
  if (!classify_operand_layout(in_strides, extents, bytes, in_layout) ||
      !classify_operand_layout(out_strides, extents, bytes, out_layout)) {
    throw LoweringException("tpp: Copy primitive '" + prim_id +
                            "' requires a tile layout with at most two axes of extent > 1 and "
                            "a unit-stride innermost axis on each operand");
  }
  const bool plain = (in_layout.unit_axis == out_layout.unit_axis);
  return compile_unary(plain ? LIBXSMM_MELTW_TYPE_UNARY_IDENTITY
                             : LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT,
                       dtype,
                       plain ? "Copy" : "Copy (transpose)",
                       prim_id,
                       in_layout.m,
                       in_layout.n,
                       in_layout.ld,
                       out_layout.ld,
                       in_idx,
                       out_idx);
}

CompiledInvocation compile_relu(const std::string& dtype,
                                int32_t in_idx,
                                int32_t out_idx,
                                const std::vector<int64_t>& in_strides,
                                const std::vector<int64_t>& out_strides,
                                const std::vector<int64_t>& extents,
                                const std::string& prim_id) {
  const int64_t bytes = dtype_bytes(dtype);
  TileLayout in_layout;
  TileLayout out_layout;
  if (!classify_operand_layout(in_strides, extents, bytes, in_layout) ||
      !classify_operand_layout(out_strides, extents, bytes, out_layout)) {
    throw LoweringException("tpp: ReLU primitive '" + prim_id +
                            "' requires a tile layout with at most two axes of extent > 1 and "
                            "a unit-stride innermost axis on each operand");
  }
  if (in_layout.unit_axis != out_layout.unit_axis) {
    throw LoweringException("tpp: ReLU primitive '" + prim_id +
                            "' requires identical unit-stride axes on input and output;"
                            " libxsmm does not provide a fused transpose-ReLU kernel");
  }
  return compile_unary(LIBXSMM_MELTW_TYPE_UNARY_RELU,
                       dtype,
                       "ReLU",
                       prim_id,
                       in_layout.m,
                       in_layout.n,
                       in_layout.ld,
                       out_layout.ld,
                       in_idx,
                       out_idx);
}

// --------------------------------------------------------------------------
// Contraction (GEMM / BRGEMM) shape planning
// --------------------------------------------------------------------------

enum class TransFlag : uint8_t { NONE = 0, TRANS_A = 1, TRANS_B = 2, TRANS_AB = 3 };

/// Everything needed to JIT one contraction kernel and execute it: the
/// libxsmm shape parameters plus the TEIR-tensor-index mapping.
struct ContractionPlan {
  // libxsmm GEMM shape.
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t lda = 0;
  int64_t ldb = 0;
  int64_t ldc = 0;
  std::string dtype;
  uint8_t flags = 0;
  bool brgemm = false;
  // BRGEMM-only: outer batch count + per-batch strides.
  int64_t br_count = 0;
  int64_t br_stride_a = 0;
  int64_t br_stride_b = 0;
  // TEIR tensor indices the dispatched kernel will read at execute time.
  int32_t a_tensor_idx = -1;
  int32_t b_tensor_idx = -1;
  int32_t c_tensor_idx = -1;
};

bool plan_libxsmm_contraction(const std::string& dtype,
                              int32_t in0_idx,
                              int32_t in1_idx,
                              int32_t out_idx,
                              int64_t m_extent,
                              int64_t n_extent,
                              int64_t k_extent,
                              int64_t m_stride_in0,
                              int64_t k_stride_in0,
                              int64_t k_stride_in1,
                              int64_t n_stride_in1,
                              int64_t m_stride_out,
                              int64_t n_stride_out,
                              bool brgemm,
                              int64_t br_count,
                              int64_t br_stride_in0,
                              int64_t br_stride_in1,
                              ContractionPlan& plan) {
  const int64_t bytes = dtype_bytes(dtype);

  // Choose a libxsmm view (mapping of TEIR (in0, in1, out) to libxsmm
  // (A, B, C)) based on the output tensor's unit-stride role axis:
  //
  // - swap view: A = in1, B = in0, C = out, M_lib = N_teir, N_lib = M_teir.
  //   Used when out's N-role axis is unit-stride. This is the canonical
  //   row-major-to-column-major view ("swap A/B, swap M/N" trick).
  //
  // - direct view: A = in0, B = in1, C = out, M_lib = M_teir, N_lib = N_teir.
  //   Used when out's M-role axis is unit-stride (column-major friendly
  //   output layouts such as the spec's `pqrs` example, where `s` is the
  //   M-role axis and is unit-stride on out).
  //
  // For each operand, we independently set a TRANS flag when the role axis
  // libxsmm wants unit-stride is not the one TEIR has unit-stride. libxsmm
  // does not support transposed output, so we cannot dispatch when neither
  // out's M nor out's N is unit-stride.

  const bool out_n_unit = (n_stride_out == bytes);
  const bool out_m_unit = (m_stride_out == bytes);
  if (!out_n_unit && !out_m_unit) {
    return false;
  }

  // View selection: pure variable assignment. The two views differ only
  // in which TEIR tensor becomes libxsmm's A vs B and which strides are
  // paired into each operand's (primary, secondary) classifier input.
  int32_t a_idx;
  int32_t b_idx;
  OperandStrides a_strides;
  OperandStrides b_strides;
  int64_t c_other_stride;
  int64_t m_lib;
  int64_t n_lib;
  int64_t br_stride_a;
  int64_t br_stride_b;

  if (out_n_unit) {
    // Swap view: A = in1, B = in0; libxsmm M_lib = N_teir.
    m_lib = n_extent;
    n_lib = m_extent;
    a_idx = in1_idx;
    b_idx = in0_idx;
    a_strides = {.primary = n_stride_in1, .secondary = k_stride_in1};
    b_strides = {.primary = k_stride_in0, .secondary = m_stride_in0};
    c_other_stride = m_stride_out;
    br_stride_a = br_stride_in1;
    br_stride_b = br_stride_in0;
  } else {
    // Direct view: A = in0, B = in1; libxsmm M_lib = M_teir.
    m_lib = m_extent;
    n_lib = n_extent;
    a_idx = in0_idx;
    b_idx = in1_idx;
    a_strides = {.primary = m_stride_in0, .secondary = k_stride_in0};
    b_strides = {.primary = k_stride_in1, .secondary = n_stride_in1};
    c_other_stride = n_stride_out;
    br_stride_a = br_stride_in0;
    br_stride_b = br_stride_in1;
  }

  // Per-operand classification: identify unit-stride axis and trans flag.
  int64_t a_ld_bytes;
  int64_t b_ld_bytes;
  bool trans_a;
  bool trans_b;
  if (!resolve_operand(a_strides, bytes, a_ld_bytes, trans_a)) {
    return false;
  }
  if (!resolve_operand(b_strides, bytes, b_ld_bytes, trans_b)) {
    return false;
  }

  if (a_ld_bytes % bytes != 0 || b_ld_bytes % bytes != 0 || c_other_stride % bytes != 0) {
    return false;
  }
  if (brgemm && (br_stride_a % bytes != 0 || br_stride_b % bytes != 0)) {
    return false;
  }

  uint8_t flags = 0;
  if (trans_a) {
    flags |= static_cast<uint8_t>(TransFlag::TRANS_A);
  }
  if (trans_b) {
    flags |= static_cast<uint8_t>(TransFlag::TRANS_B);
  }

  plan = ContractionPlan{};
  plan.dtype = dtype;
  plan.brgemm = brgemm;
  plan.m = m_lib;
  plan.n = n_lib;
  plan.k = k_extent;
  plan.lda = a_ld_bytes / bytes;
  plan.ldb = b_ld_bytes / bytes;
  plan.ldc = c_other_stride / bytes;
  plan.flags = flags;
  plan.a_tensor_idx = a_idx;
  plan.b_tensor_idx = b_idx;
  plan.c_tensor_idx = out_idx;
  if (brgemm) {
    plan.br_count = br_count;
    plan.br_stride_a = br_stride_a;
    plan.br_stride_b = br_stride_b;
  }
  return true;
}

#endif // ETOPS_LIBXSMM_AVAILABLE

// --------------------------------------------------------------------------
// Primitive entry points
// --------------------------------------------------------------------------

CompiledInvocation tpp_zero(const Primitive& prim, const Teir& teir) {
#if ETOPS_LIBXSMM_AVAILABLE
  if (teir.tensors.empty()) {
    throw LoweringException("tpp: Zero primitive '" + prim.id + "' requires at least one tensor");
  }
  const int32_t out_idx = static_cast<int32_t>(teir.tensors.size() - 1);
  std::vector<int64_t> in_strides;
  std::vector<int64_t> out_strides;
  std::vector<int64_t> extents;
  collect_role_strides(prim, teir, "M", out_idx, out_idx, in_strides, out_strides, extents);
  collect_role_strides(prim, teir, "N", out_idx, out_idx, in_strides, out_strides, extents);
  const std::string dtype = data_type_of(prim);
  return compile_zero(dtype, out_idx, out_strides, extents, prim.id);
#else
  (void)teir;
  throw LoweringException("tpp: Zero primitive '" + prim.id +
                          "' requires libxsmm, which is not compiled into this build");
#endif
}

CompiledInvocation tpp_copy(const Primitive& prim, const Teir& teir) {
#if ETOPS_LIBXSMM_AVAILABLE
  if (teir.tensors.size() < 2) {
    throw LoweringException("tpp: copy requires 2 tensors");
  }
  const int32_t in_idx = 0;
  const int32_t out_idx = static_cast<int32_t>(teir.tensors.size() - 1);
  std::vector<int64_t> in_strides;
  std::vector<int64_t> out_strides;
  std::vector<int64_t> extents;
  collect_role_strides(prim, teir, "M", in_idx, out_idx, in_strides, out_strides, extents);
  collect_role_strides(prim, teir, "N", in_idx, out_idx, in_strides, out_strides, extents);
  const std::string dtype = data_type_of(prim);
  return compile_copy(dtype, in_idx, out_idx, in_strides, out_strides, extents, prim.id);
#else
  (void)teir;
  throw LoweringException("tpp: Copy primitive '" + prim.id +
                          "' requires libxsmm, which is not compiled into this build");
#endif
}

CompiledInvocation tpp_relu(const Primitive& prim, const Teir& teir) {
#if ETOPS_LIBXSMM_AVAILABLE
  if (teir.tensors.size() < 2) {
    throw LoweringException("tpp: relu requires 2 tensors");
  }
  const int32_t in_idx = 0;
  const int32_t out_idx = static_cast<int32_t>(teir.tensors.size() - 1);
  std::vector<int64_t> in_strides;
  std::vector<int64_t> out_strides;
  std::vector<int64_t> extents;
  collect_role_strides(prim, teir, "M", in_idx, out_idx, in_strides, out_strides, extents);
  collect_role_strides(prim, teir, "N", in_idx, out_idx, in_strides, out_strides, extents);
  const std::string dtype = data_type_of(prim);
  return compile_relu(dtype, in_idx, out_idx, in_strides, out_strides, extents, prim.id);
#else
  (void)teir;
  throw LoweringException("tpp: ReLU primitive '" + prim.id +
                          "' requires libxsmm, which is not compiled into this build");
#endif
}

#if ETOPS_LIBXSMM_AVAILABLE
CompiledInvocation finish_contraction(const ContractionPlan& plan) {
  ensure_libxsmm_init();
  const libxsmm_datatype dt = to_libxsmm_dtype(plan.dtype);
  if (dt == LIBXSMM_DATATYPE_UNSUPPORTED) {
    return CompiledInvocation{};
  }
  if (!fits_in_blasint(plan.m) || !fits_in_blasint(plan.n) || !fits_in_blasint(plan.k) ||
      !fits_in_blasint(plan.lda) || !fits_in_blasint(plan.ldb) || !fits_in_blasint(plan.ldc) ||
      !fits_in_blasint(plan.br_stride_a) || !fits_in_blasint(plan.br_stride_b)) {
    return CompiledInvocation{};
  }
  libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(static_cast<libxsmm_blasint>(plan.m),
                                                       static_cast<libxsmm_blasint>(plan.n),
                                                       static_cast<libxsmm_blasint>(plan.k),
                                                       static_cast<libxsmm_blasint>(plan.lda),
                                                       static_cast<libxsmm_blasint>(plan.ldb),
                                                       static_cast<libxsmm_blasint>(plan.ldc),
                                                       dt,
                                                       dt,
                                                       dt,
                                                       dt);
  libxsmm_bitfield gemm_flags = LIBXSMM_GEMM_FLAG_NONE;
  if ((plan.flags & static_cast<uint8_t>(TransFlag::TRANS_A)) != 0) {
    gemm_flags |= LIBXSMM_GEMM_FLAG_TRANS_A;
  }
  if ((plan.flags & static_cast<uint8_t>(TransFlag::TRANS_B)) != 0) {
    gemm_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  // libxsmm dedupes by descriptor in a thread-local L1 + lock-free global
  // registry, so no wrapper cache is needed.
  libxsmm_gemmfunction fn = nullptr;
  if (plan.brgemm) {
    libxsmm_gemm_batch_reduce_config br =
        libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE,
                                                static_cast<libxsmm_blasint>(plan.br_stride_a),
                                                static_cast<libxsmm_blasint>(plan.br_stride_b),
                                                /*br_unroll_hint=*/0);
    fn = libxsmm_dispatch_brgemm(shape, gemm_flags, /*prefetch_flags=*/0, br);
  } else {
    fn = libxsmm_dispatch_gemm(shape, gemm_flags, /*prefetch_flags=*/0);
  }
  if (fn == nullptr) {
    return CompiledInvocation{};
  }

  libxsmm_kernel_info info{};
  if (libxsmm_get_kernel_info(reinterpret_cast<const void*>(fn), &info) == 0 &&
      info.is_reference_kernel != 0) {
    return CompiledInvocation{};
  }
  auto* state = new ContractionState{.fn = fn,
                                     .a_tensor_idx = plan.a_tensor_idx,
                                     .b_tensor_idx = plan.b_tensor_idx,
                                     .c_tensor_idx = plan.c_tensor_idx,
                                     .br_count = plan.br_count};
  return CompiledInvocation{
      .kernel = &contraction_kernel,
      .state = {state, &state_deleter<ContractionState>},
  };
}
#endif

CompiledInvocation tpp_contraction(const Primitive& prim, const Teir& teir) {
#if ETOPS_LIBXSMM_AVAILABLE
  if (teir.tensors.size() < 3) {
    throw LoweringException("tpp: contraction requires 3 tensors");
  }
  const int32_t in0_idx = 0;
  const int32_t in1_idx = 1;
  const int32_t out_idx = static_cast<int32_t>(teir.tensors.size() - 1);
  const auto& m_axes = role(prim, "M");
  const auto& n_axes = role(prim, "N");
  const auto& k_axes = role(prim, "K");

  std::vector<int64_t> m_strides_in0;
  std::vector<int64_t> m_strides_out;
  std::vector<int64_t> m_extents;
  for (const auto& a : m_axes) {
    const Axis& ax = axis_by_id(teir, a);
    m_strides_in0.push_back(stride_of(ax, in0_idx));
    m_strides_out.push_back(stride_of(ax, out_idx));
    m_extents.push_back(ax.extent);
  }
  std::vector<int64_t> n_strides_in1;
  std::vector<int64_t> n_strides_out;
  std::vector<int64_t> n_extents;
  for (const auto& a : n_axes) {
    const Axis& ax = axis_by_id(teir, a);
    n_strides_in1.push_back(stride_of(ax, in1_idx));
    n_strides_out.push_back(stride_of(ax, out_idx));
    n_extents.push_back(ax.extent);
  }
  std::vector<int64_t> k_strides_in0;
  std::vector<int64_t> k_strides_in1;
  std::vector<int64_t> k_extents;
  for (const auto& a : k_axes) {
    const Axis& ax = axis_by_id(teir, a);
    k_strides_in0.push_back(stride_of(ax, in0_idx));
    k_strides_in1.push_back(stride_of(ax, in1_idx));
    k_extents.push_back(ax.extent);
  }

  const std::string dtype = data_type_of(prim);

  if (m_extents.size() == 1 && n_extents.size() == 1 && k_extents.size() == 1) {
    ContractionPlan plan;
    if (plan_libxsmm_contraction(dtype,
                                 in0_idx,
                                 in1_idx,
                                 out_idx,
                                 m_extents[0],
                                 n_extents[0],
                                 k_extents[0],
                                 m_strides_in0[0],
                                 k_strides_in0[0],
                                 k_strides_in1[0],
                                 n_strides_in1[0],
                                 m_strides_out[0],
                                 n_strides_out[0],
                                 /*brgemm=*/false,
                                 /*br_count=*/0,
                                 /*br_stride_in0=*/0,
                                 /*br_stride_in1=*/0,
                                 plan)) {
      CompiledInvocation inv = finish_contraction(plan);
      if (inv.kernel != nullptr) {
        return inv;
      }
    }
    throw LoweringException("tpp: libxsmm could not dispatch a GEMM for primitive '" + prim.id +
                            "' (dtype=" + dtype +
                            "); the optimizer must arrange the role axes so that one of in0's "
                            "{M, K}, one of in1's {K, N}, and one of out's {M, N} role axes is "
                            "unit-stride and the remaining role axis is element-aligned");
  }
  if (m_extents.size() == 1 && n_extents.size() == 1 && k_extents.size() == 2) {
    // The outermost K is the batch-reduce dimension; the innermost K plays
    // the GEMM K role. Strides on the outer K become the BRGEMM strides.
    ContractionPlan plan;
    if (plan_libxsmm_contraction(dtype,
                                 in0_idx,
                                 in1_idx,
                                 out_idx,
                                 m_extents[0],
                                 n_extents[0],
                                 k_extents[1],
                                 m_strides_in0[0],
                                 k_strides_in0[1],
                                 k_strides_in1[1],
                                 n_strides_in1[0],
                                 m_strides_out[0],
                                 n_strides_out[0],
                                 /*brgemm=*/true,
                                 /*br_count=*/k_extents[0],
                                 /*br_stride_in0=*/k_strides_in0[0],
                                 /*br_stride_in1=*/k_strides_in1[0],
                                 plan)) {
      CompiledInvocation inv = finish_contraction(plan);
      if (inv.kernel != nullptr) {
        return inv;
      }
    }
    throw LoweringException("tpp: libxsmm could not dispatch a BRGEMM for primitive '" + prim.id +
                            "' (dtype=" + dtype + ")");
  }
  throw LoweringException("tpp: contraction primitive '" + prim.id +
                          "' has role cardinalities (M=" + std::to_string(m_extents.size()) +
                          ", N=" + std::to_string(n_extents.size()) +
                          ", K=" + std::to_string(k_extents.size()) +
                          ") outside the GEMM (1, 1, 1) and BRGEMM (1, 1, 2) shapes libxsmm"
                          " supports; run the TPP optimization pipeline before lowering or"
                          " choose a different backend");
#else
  (void)teir;
  throw LoweringException("tpp: contraction primitive '" + prim.id +
                          "' requires libxsmm, which is not compiled into this build");
#endif
}

} // namespace

void register_tpp_primitives() {
  register_primitive("tpp", "Zero", tpp_zero);
  register_primitive("tpp", "Copy", tpp_copy);
  register_primitive("tpp", "ReLU", tpp_relu);
  register_primitive("tpp", "Contraction", tpp_contraction);
}

} // namespace teir::tpp
