/// BLAS backend primitive lowerings.
///
/// Unary primitives (Zero / Copy / ReLU) take stack-allocated per-tile
/// loops with vendor fast paths: contiguous 1-D Copy goes through
/// `cblas_*copy`, contiguous 1-D Zero through `memset`, and a 2-D Copy
/// whose access pattern matches `cblas_*omatcopy` is dispatched there
/// when the vendor exposes it. ReLU stays in-house. Contractions
/// dispatch through `cblas_sgemm` / `cblas_dgemm` for full GEMM and
/// through `cblas_sgemv` / `cblas_dgemv` for the BLAS-2 GEMV shape;
/// BRGEMM is unsupported.
///
/// Per-vendor single-threaded enforcement runs once on module init so the
/// `etops` scheduler owns parallelism while each BLAS call runs sequentially.

#include "internal/primitive_utils.h"
#include "teir/config.h"
#include "teir/exception.h"
#include "teir/ir.h"
#include "teir/primitive.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#if ETOPS_BLAS_AVAILABLE

#  if TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_OPENBLAS
#    include <cblas.h>
#  elif TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_MKL
#    include <mkl_cblas.h>
#    include <mkl_service.h>
#  elif TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_ACCELERATE
#    include <Accelerate/Accelerate.h>
#  else
// Best-effort: a generic system that exposes a cblas-compatible API via
// <cblas.h>. If the header is missing the build leaves
// ETOPS_BLAS_AVAILABLE = 0 and this file's body is never compiled.
#    include <cblas.h>
#  endif

#endif

namespace teir::blas {

namespace {

constexpr const char* BACKEND = "blas";

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

/// True iff ``value`` is representable as ``int``. cblas takes ``int`` for
/// sizes and strides, so values that would silently truncate at the dispatch
/// boundary are caught at lowering instead of producing wrong-shape runtime output.
inline bool fits_in_int(int64_t value) noexcept {
  return value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

// --------------------------------------------------------------------------
// Unary kernels (Zero / Copy / ReLU).
//
// Two paths per primitive: a fast rank-1 path (`memset` / `memcpy` /
// `cblas_*copy`) plus an `omatcopy` fast path for transposed rank-2
// copies, then a generic N-D walk for everything else. Tile ranks beyond
// ``MAX_UNARY_AXES`` are rejected.
// --------------------------------------------------------------------------

constexpr std::size_t MAX_UNARY_AXES = 8;

template <typename T>
void zero_1d(char* base, int64_t stride0, int64_t extent0) {
  if (stride0 == static_cast<int64_t>(sizeof(T))) {
    std::memset(base, 0, static_cast<std::size_t>(extent0) * sizeof(T));
    return;
  }
  for (int64_t i = 0; i < extent0; ++i) {
    *reinterpret_cast<T*>(base + i * stride0) = T(0);
  }
}

template <typename T>
void copy_1d(
    char* in_base, char* out_base, int64_t in_stride, int64_t out_stride, int64_t extent0) {
  const int64_t elem = static_cast<int64_t>(sizeof(T));
  if (in_stride == elem && out_stride == elem) {
    std::memcpy(out_base, in_base, static_cast<std::size_t>(extent0) * sizeof(T));
    return;
  }
#if ETOPS_BLAS_AVAILABLE
  if constexpr (std::is_same_v<T, float>) {
    cblas_scopy(static_cast<int>(extent0),
                reinterpret_cast<const float*>(in_base),
                static_cast<int>(in_stride / elem),
                reinterpret_cast<float*>(out_base),
                static_cast<int>(out_stride / elem));
    return;
  }
  if constexpr (std::is_same_v<T, double>) {
    cblas_dcopy(static_cast<int>(extent0),
                reinterpret_cast<const double*>(in_base),
                static_cast<int>(in_stride / elem),
                reinterpret_cast<double*>(out_base),
                static_cast<int>(out_stride / elem));
    return;
  }
#endif
  for (int64_t i = 0; i < extent0; ++i) {
    *reinterpret_cast<T*>(out_base + i * out_stride) =
        *reinterpret_cast<const T*>(in_base + i * in_stride);
  }
}

template <typename T>
bool try_omatcopy_2d(char* in_base,
                     char* out_base,
                     int64_t in_stride0,
                     int64_t in_stride1,
                     int64_t out_stride0,
                     int64_t out_stride1,
                     int64_t extent0,
                     int64_t extent1) {
#if ETOPS_BLAS_AVAILABLE && TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_OPENBLAS
  const int64_t elem = static_cast<int64_t>(sizeof(T));
  const bool in_outer_unit = in_stride0 == elem;
  const bool in_inner_unit = in_stride1 == elem;
  const bool out_inner_unit = out_stride1 == elem;
  if (out_inner_unit && in_inner_unit) {
    if (in_stride0 % elem != 0 || out_stride0 % elem != 0) {
      return false;
    }
    if constexpr (std::is_same_v<T, float>) {
      cblas_somatcopy(CblasRowMajor,
                      CblasNoTrans,
                      static_cast<size_t>(extent0),
                      static_cast<size_t>(extent1),
                      1.0f,
                      reinterpret_cast<const float*>(in_base),
                      static_cast<size_t>(in_stride0 / elem),
                      reinterpret_cast<float*>(out_base),
                      static_cast<size_t>(out_stride0 / elem));
      return true;
    }
    if constexpr (std::is_same_v<T, double>) {
      cblas_domatcopy(CblasRowMajor,
                      CblasNoTrans,
                      static_cast<size_t>(extent0),
                      static_cast<size_t>(extent1),
                      1.0,
                      reinterpret_cast<const double*>(in_base),
                      static_cast<size_t>(in_stride0 / elem),
                      reinterpret_cast<double*>(out_base),
                      static_cast<size_t>(out_stride0 / elem));
      return true;
    }
  }
  if (out_inner_unit && in_outer_unit) {
    if (in_stride1 % elem != 0 || out_stride0 % elem != 0) {
      return false;
    }
    if constexpr (std::is_same_v<T, float>) {
      cblas_somatcopy(CblasRowMajor,
                      CblasTrans,
                      static_cast<size_t>(extent1),
                      static_cast<size_t>(extent0),
                      1.0f,
                      reinterpret_cast<const float*>(in_base),
                      static_cast<size_t>(in_stride1 / elem),
                      reinterpret_cast<float*>(out_base),
                      static_cast<size_t>(out_stride0 / elem));
      return true;
    }
    if constexpr (std::is_same_v<T, double>) {
      cblas_domatcopy(CblasRowMajor,
                      CblasTrans,
                      static_cast<size_t>(extent1),
                      static_cast<size_t>(extent0),
                      1.0,
                      reinterpret_cast<const double*>(in_base),
                      static_cast<size_t>(in_stride1 / elem),
                      reinterpret_cast<double*>(out_base),
                      static_cast<size_t>(out_stride0 / elem));
      return true;
    }
  }
#else
  (void)in_base;
  (void)out_base;
  (void)in_stride0;
  (void)in_stride1;
  (void)out_stride0;
  (void)out_stride1;
  (void)extent0;
  (void)extent1;
#endif
  return false;
}

/// Increment a row-major multi-index. Returns true while ``idx`` is still
/// in-bounds; returns false when the iteration has completed.
inline bool
advance(std::array<int64_t, MAX_UNARY_AXES>& idx, const int64_t* extents, std::size_t rank) {
  std::size_t dim = rank;
  while (dim > 0) {
    --dim;
    if (++idx[dim] < extents[dim]) {
      return true;
    }
    idx[dim] = 0;
    if (dim == 0) {
      return false;
    }
  }
  return false;
}

template <typename T, typename Body>
void walk_unary(
    char* base, const int64_t* strides, const int64_t* extents, std::size_t rank, Body body) {
  std::array<int64_t, MAX_UNARY_AXES> idx{};
  while (true) {
    int64_t off = 0;
    for (std::size_t k = 0; k < rank; ++k) {
      off += idx[k] * strides[k];
    }
    body(base + off);
    if (rank == 0 || !advance(idx, extents, rank)) {
      return;
    }
  }
}

template <typename T, typename Body>
void walk_unary_inout(char* in_base,
                      char* out_base,
                      const int64_t* in_strides,
                      const int64_t* out_strides,
                      const int64_t* extents,
                      std::size_t rank,
                      Body body) {
  std::array<int64_t, MAX_UNARY_AXES> idx{};
  while (true) {
    int64_t in_off = 0;
    int64_t out_off = 0;
    for (std::size_t k = 0; k < rank; ++k) {
      in_off += idx[k] * in_strides[k];
      out_off += idx[k] * out_strides[k];
    }
    body(in_base + in_off, out_base + out_off);
    if (rank == 0 || !advance(idx, extents, rank)) {
      return;
    }
  }
}

template <typename T>
void scalar_zero(char* base,
                 const std::vector<int64_t>& byte_strides,
                 const std::vector<int64_t>& extents) {
  const std::size_t rank = extents.size();
  if (rank == 1) {
    zero_1d<T>(base, byte_strides[0], extents[0]);
    return;
  }
  if (rank > MAX_UNARY_AXES) {
    throw LoweringException("blas: zero tile rank exceeds in-line limit " +
                            std::to_string(MAX_UNARY_AXES));
  }
  walk_unary<T>(base, byte_strides.data(), extents.data(), rank, [](char* ptr) {
    *reinterpret_cast<T*>(ptr) = T(0);
  });
}

template <typename T>
void scalar_copy(char* in_base,
                 char* out_base,
                 const std::vector<int64_t>& in_strides,
                 const std::vector<int64_t>& out_strides,
                 const std::vector<int64_t>& extents) {
  const std::size_t rank = extents.size();
  if (rank == 1) {
    copy_1d<T>(in_base, out_base, in_strides[0], out_strides[0], extents[0]);
    return;
  }
  if (rank == 2 && try_omatcopy_2d<T>(in_base,
                                      out_base,
                                      in_strides[0],
                                      in_strides[1],
                                      out_strides[0],
                                      out_strides[1],
                                      extents[0],
                                      extents[1])) {
    return;
  }
  if (rank > MAX_UNARY_AXES) {
    throw LoweringException("blas: copy tile rank exceeds in-line limit " +
                            std::to_string(MAX_UNARY_AXES));
  }
  walk_unary_inout<T>(in_base,
                      out_base,
                      in_strides.data(),
                      out_strides.data(),
                      extents.data(),
                      rank,
                      [](char* in_ptr, char* out_ptr) {
                        *reinterpret_cast<T*>(out_ptr) = *reinterpret_cast<const T*>(in_ptr);
                      });
}

template <typename T>
void scalar_relu(char* in_base,
                 char* out_base,
                 const std::vector<int64_t>& in_strides,
                 const std::vector<int64_t>& out_strides,
                 const std::vector<int64_t>& extents) {
  const std::size_t rank = extents.size();
  if (rank > MAX_UNARY_AXES) {
    throw LoweringException("blas: relu tile rank exceeds in-line limit " +
                            std::to_string(MAX_UNARY_AXES));
  }
  walk_unary_inout<T>(in_base,
                      out_base,
                      in_strides.data(),
                      out_strides.data(),
                      extents.data(),
                      rank,
                      [](char* in_ptr, char* out_ptr) {
                        T x = *reinterpret_cast<const T*>(in_ptr);
                        *reinterpret_cast<T*>(out_ptr) = x > T(0) ? x : T(0);
                      });
}

// --------------------------------------------------------------------------
// Contraction
// --------------------------------------------------------------------------

#if ETOPS_BLAS_AVAILABLE

// CBLAS uses 32-bit ints for matrix dimensions and leading dimensions on
// every supported vendor (OpenBLAS, MKL ILP-LP, Accelerate). Reject calls
// that would silently truncate to negative values before we hand them to
// the vendor routine.
void check_blas_int_range(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  constexpr int64_t MAX_INT = std::numeric_limits<int>::max();
  if (m > MAX_INT || n > MAX_INT || k > MAX_INT || lda > MAX_INT || ldb > MAX_INT ||
      ldc > MAX_INT) {
    throw LoweringException("blas: GEMM dimensions exceed INT_MAX (CBLAS uses 32-bit ints)");
  }
}

void cblas_gemm_f32(CBLAS_TRANSPOSE trans_a,
                    CBLAS_TRANSPOSE trans_b,
                    int64_t m,
                    int64_t n,
                    int64_t k,
                    const float* a,
                    int64_t lda,
                    const float* b,
                    int64_t ldb,
                    float* c,
                    int64_t ldc) {
  check_blas_int_range(m, n, k, lda, ldb, ldc);
  cblas_sgemm(CblasRowMajor,
              trans_a,
              trans_b,
              static_cast<int>(m),
              static_cast<int>(n),
              static_cast<int>(k),
              1.0f,
              a,
              static_cast<int>(lda),
              b,
              static_cast<int>(ldb),
              1.0f,
              c,
              static_cast<int>(ldc));
}

void cblas_gemm_f64(CBLAS_TRANSPOSE trans_a,
                    CBLAS_TRANSPOSE trans_b,
                    int64_t m,
                    int64_t n,
                    int64_t k,
                    const double* a,
                    int64_t lda,
                    const double* b,
                    int64_t ldb,
                    double* c,
                    int64_t ldc) {
  check_blas_int_range(m, n, k, lda, ldb, ldc);
  cblas_dgemm(CblasRowMajor,
              trans_a,
              trans_b,
              static_cast<int>(m),
              static_cast<int>(n),
              static_cast<int>(k),
              1.0,
              a,
              static_cast<int>(lda),
              b,
              static_cast<int>(ldb),
              1.0,
              c,
              static_cast<int>(ldc));
}

#endif

// --------------------------------------------------------------------------
// Primitive entry points
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Per-invocation state + kernel trampolines
// --------------------------------------------------------------------------

struct BlasUnaryState {
  int32_t in_tensor_idx;
  int32_t out_tensor_idx;
  std::vector<int64_t> in_strides;
  std::vector<int64_t> out_strides;
  std::vector<int64_t> extents;
};

struct BlasContractionState {
  int32_t a_tensor_idx;
  int32_t b_tensor_idx;
  int32_t c_tensor_idx;
  CBLAS_TRANSPOSE trans_a;
  CBLAS_TRANSPOSE trans_b;
  int64_t m_lib;
  int64_t n_lib;
  int64_t k_extent;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
};

template <typename T>
void state_deleter(void* p) {
  delete static_cast<T*>(p);
}

template <typename T>
void blas_zero_kernel(const TileAddresses& tiles, const void* state) {
  const auto* s = static_cast<const BlasUnaryState*>(state);
  scalar_zero<T>(static_cast<char*>(tiles.addrs[static_cast<std::size_t>(s->out_tensor_idx)]),
                 s->out_strides,
                 s->extents);
}

template <typename T>
void blas_copy_kernel(const TileAddresses& tiles, const void* state) {
  const auto* s = static_cast<const BlasUnaryState*>(state);
  scalar_copy<T>(static_cast<char*>(tiles.addrs[static_cast<std::size_t>(s->in_tensor_idx)]),
                 static_cast<char*>(tiles.addrs[static_cast<std::size_t>(s->out_tensor_idx)]),
                 s->in_strides,
                 s->out_strides,
                 s->extents);
}

template <typename T>
void blas_relu_kernel(const TileAddresses& tiles, const void* state) {
  const auto* s = static_cast<const BlasUnaryState*>(state);
  scalar_relu<T>(static_cast<char*>(tiles.addrs[static_cast<std::size_t>(s->in_tensor_idx)]),
                 static_cast<char*>(tiles.addrs[static_cast<std::size_t>(s->out_tensor_idx)]),
                 s->in_strides,
                 s->out_strides,
                 s->extents);
}

#if ETOPS_BLAS_AVAILABLE
void blas_gemm_f32_kernel(const TileAddresses& tiles, const void* state) {
  const auto* s = static_cast<const BlasContractionState*>(state);
  cblas_gemm_f32(s->trans_a,
                 s->trans_b,
                 s->m_lib,
                 s->n_lib,
                 s->k_extent,
                 static_cast<const float*>(tiles.addrs[static_cast<std::size_t>(s->a_tensor_idx)]),
                 s->lda,
                 static_cast<const float*>(tiles.addrs[static_cast<std::size_t>(s->b_tensor_idx)]),
                 s->ldb,
                 static_cast<float*>(tiles.addrs[static_cast<std::size_t>(s->c_tensor_idx)]),
                 s->ldc);
}

void blas_gemm_f64_kernel(const TileAddresses& tiles, const void* state) {
  const auto* s = static_cast<const BlasContractionState*>(state);
  cblas_gemm_f64(s->trans_a,
                 s->trans_b,
                 s->m_lib,
                 s->n_lib,
                 s->k_extent,
                 static_cast<const double*>(tiles.addrs[static_cast<std::size_t>(s->a_tensor_idx)]),
                 s->lda,
                 static_cast<const double*>(tiles.addrs[static_cast<std::size_t>(s->b_tensor_idx)]),
                 s->ldb,
                 static_cast<double*>(tiles.addrs[static_cast<std::size_t>(s->c_tensor_idx)]),
                 s->ldc);
}
#endif

CompiledInvocation blas_zero(const Primitive& prim, const Teir& teir) {
  if (teir.tensors.empty()) {
    throw LoweringException("blas: Zero primitive '" + prim.id + "' requires at least one tensor");
  }
  const int32_t out_idx = static_cast<int32_t>(teir.tensors.size() - 1);
  const auto& m_axes = role(prim, "M");
  const auto& n_axes = role(prim, "N");
  auto state = std::make_unique<BlasUnaryState>();
  state->in_tensor_idx = out_idx;
  state->out_tensor_idx = out_idx;
  for (const auto& a : m_axes) {
    const Axis& ax = axis_by_id(teir, a);
    state->out_strides.push_back(stride_of(ax, out_idx));
    state->extents.push_back(ax.extent);
  }
  for (const auto& a : n_axes) {
    const Axis& ax = axis_by_id(teir, a);
    state->out_strides.push_back(stride_of(ax, out_idx));
    state->extents.push_back(ax.extent);
  }
  const std::string dtype = data_type_of(prim);
  KernelFn kernel;
  if (dtype == "f32") {
    kernel = &blas_zero_kernel<float>;
  } else if (dtype == "f64") {
    kernel = &blas_zero_kernel<double>;
  } else {
    throw LoweringException("blas: zero unsupported dtype " + dtype);
  }
  return CompiledInvocation{
      kernel,
      {state.release(), &state_deleter<BlasUnaryState>},
  };
}

std::unique_ptr<BlasUnaryState>
build_unary_state(const Primitive& prim, const Teir& teir, int32_t in_idx, int32_t out_idx) {
  auto state = std::make_unique<BlasUnaryState>();
  state->in_tensor_idx = in_idx;
  state->out_tensor_idx = out_idx;
  for (const auto& a : role(prim, "M")) {
    const Axis& ax = axis_by_id(teir, a);
    state->in_strides.push_back(stride_of(ax, in_idx));
    state->out_strides.push_back(stride_of(ax, out_idx));
    state->extents.push_back(ax.extent);
  }
  for (const auto& a : role(prim, "N")) {
    const Axis& ax = axis_by_id(teir, a);
    state->in_strides.push_back(stride_of(ax, in_idx));
    state->out_strides.push_back(stride_of(ax, out_idx));
    state->extents.push_back(ax.extent);
  }
  return state;
}

CompiledInvocation blas_copy(const Primitive& prim, const Teir& teir) {
  if (teir.tensors.size() < 2) {
    throw LoweringException("blas: copy requires 2 tensors");
  }
  const int32_t in_idx = 0;
  const int32_t out_idx = static_cast<int32_t>(teir.tensors.size() - 1);
  auto state = build_unary_state(prim, teir, in_idx, out_idx);
  const std::string dtype = data_type_of(prim);

  if (state->extents.size() == 1) {
    const int64_t elem = dtype_bytes(dtype);
    const int64_t extent = state->extents[0];
    const int64_t in_stride = state->in_strides[0];
    const int64_t out_stride = state->out_strides[0];
    const bool unit_unit = in_stride == elem && out_stride == elem;
    if (!unit_unit) {
      const bool extent_ok = fits_in_int(extent);
      const bool in_ok = in_stride % elem != 0 || fits_in_int(in_stride / elem);
      const bool out_ok = out_stride % elem != 0 || fits_in_int(out_stride / elem);
      if (!extent_ok || !in_ok || !out_ok) {
        throw LoweringException("blas: copy primitive '" + prim.id +
                                "' has rank-1 extent/stride exceeding cblas int range (extent=" +
                                std::to_string(extent) +
                                ", in_stride=" + std::to_string(in_stride) +
                                ", out_stride=" + std::to_string(out_stride) + ")");
      }
    }
  }

  KernelFn kernel;
  if (dtype == "f32") {
    kernel = &blas_copy_kernel<float>;
  } else if (dtype == "f64") {
    kernel = &blas_copy_kernel<double>;
  } else {
    throw LoweringException("blas: copy unsupported dtype " + dtype);
  }
  return CompiledInvocation{
      kernel,
      {state.release(), &state_deleter<BlasUnaryState>},
  };
}

CompiledInvocation blas_relu(const Primitive& prim, const Teir& teir) {
  if (teir.tensors.size() < 2) {
    throw LoweringException("blas: relu requires 2 tensors");
  }
  const int32_t in_idx = 0;
  const int32_t out_idx = static_cast<int32_t>(teir.tensors.size() - 1);
  auto state = build_unary_state(prim, teir, in_idx, out_idx);
  const std::string dtype = data_type_of(prim);
  KernelFn kernel;
  if (dtype == "f32") {
    kernel = &blas_relu_kernel<float>;
  } else if (dtype == "f64") {
    kernel = &blas_relu_kernel<double>;
  } else {
    throw LoweringException("blas: relu unsupported dtype " + dtype);
  }
  return CompiledInvocation{
      kernel,
      {state.release(), &state_deleter<BlasUnaryState>},
  };
}

CompiledInvocation blas_contraction(const Primitive& prim, const Teir& teir) {
  if (teir.tensors.size() < 3) {
    throw LoweringException("blas: contraction requires 3 tensors");
  }
  const int32_t in0_idx = 0;
  const int32_t in1_idx = 1;
  const int32_t out_idx = static_cast<int32_t>(teir.tensors.size() - 1);
  const auto& m_axes = role(prim, "M");
  const auto& n_axes = role(prim, "N");
  const auto& k_axes = role(prim, "K");

  std::vector<int64_t> m_strides_in0, m_strides_out, m_extents;
  for (const auto& a : m_axes) {
    const Axis& ax = axis_by_id(teir, a);
    m_strides_in0.push_back(stride_of(ax, in0_idx));
    m_strides_out.push_back(stride_of(ax, out_idx));
    m_extents.push_back(ax.extent);
  }
  std::vector<int64_t> n_strides_in1, n_strides_out, n_extents;
  for (const auto& a : n_axes) {
    const Axis& ax = axis_by_id(teir, a);
    n_strides_in1.push_back(stride_of(ax, in1_idx));
    n_strides_out.push_back(stride_of(ax, out_idx));
    n_extents.push_back(ax.extent);
  }
  std::vector<int64_t> k_strides_in0, k_strides_in1, k_extents;
  for (const auto& a : k_axes) {
    const Axis& ax = axis_by_id(teir, a);
    k_strides_in0.push_back(stride_of(ax, in0_idx));
    k_strides_in1.push_back(stride_of(ax, in1_idx));
    k_extents.push_back(ax.extent);
  }
  const std::string dtype = data_type_of(prim);

#if ETOPS_BLAS_AVAILABLE
  if (k_extents.size() >= 2) {
    throw LoweringException("blas: BRGEMM (Contraction with two K role axes) is not supported"
                            " by the BLAS backend; use the TPP backend, or have the optimization"
                            " pipeline collapse the outer K loop into the schedule");
  }
  if (m_extents.size() != 1 || n_extents.size() != 1 || k_extents.size() != 1) {
    throw LoweringException("blas: contraction primitive '" + prim.id +
                            "' has role cardinalities (M=" + std::to_string(m_extents.size()) +
                            ", N=" + std::to_string(n_extents.size()) +
                            ", K=" + std::to_string(k_extents.size()) +
                            "); BLAS dispatch requires exactly one role axis per M, N, K."
                            " Run the BLAS optimization pipeline before lowering.");
  }
  const int64_t m_extent = m_extents[0];
  const int64_t n_extent = n_extents[0];
  const int64_t k_extent = k_extents[0];
  const int64_t bytes = dtype_bytes(dtype);
  const int64_t m_stride_in0 = m_strides_in0[0];
  const int64_t k_stride_in0 = k_strides_in0[0];
  const int64_t k_stride_in1 = k_strides_in1[0];
  const int64_t n_stride_in1 = n_strides_in1[0];
  const int64_t m_stride_out = m_strides_out[0];
  const int64_t n_stride_out = n_strides_out[0];

  // CBLAS row-major requires the output's contiguous (cols) dimension to
  // be unit-stride. We pick the (M, N) ↔ (cblas-M, cblas-N) mapping based
  // on which output axis carries the unit stride:
  //   - out's TEIR-N is unit-stride: direct mapping (M_cblas = M_teir).
  //   - out's TEIR-M is unit-stride: swapped mapping (M_cblas = N_teir).
  // Per operand we then choose TRANS_A / TRANS_B by examining which of
  // its role axes is unit-stride.
  const bool n_on_out_unit = (n_stride_out == bytes);
  const bool m_on_out_unit = (m_stride_out == bytes);
  if (!n_on_out_unit && !m_on_out_unit) {
    throw LoweringException("blas: contraction primitive '" + prim.id +
                            "' requires one of out's {M, N} role axes to be unit-stride;"
                            " got M-stride-out=" +
                            std::to_string(m_stride_out) +
                            ", N-stride-out=" + std::to_string(n_stride_out));
  }

  // View selection: pure variable assignment. Direct mapping uses
  // (A=in0, B=in1, M_cblas=M_teir); swap mapping uses
  // (A=in1, B=in0, M_cblas=N_teir).
  int32_t a_idx, b_idx;
  int64_t m_lib, n_lib, ldc_b;
  OperandStrides a_strides, b_strides;
  // Diagnostic labels: which TEIR tensor and which axis pair feed each
  // operand, used in the error path when resolve_operand fails.
  std::string a_tensor_label, b_tensor_label;
  std::string a_axes_label, b_axes_label;

  if (n_on_out_unit) {
    m_lib = m_extent;
    n_lib = n_extent;
    ldc_b = m_stride_out;
    a_idx = in0_idx;
    b_idx = in1_idx;
    a_strides = {k_stride_in0, m_stride_in0};
    b_strides = {n_stride_in1, k_stride_in1};
    a_tensor_label = "in0";
    b_tensor_label = "in1";
    a_axes_label = "{K, M}";
    b_axes_label = "{N, K}";
  } else {
    m_lib = n_extent;
    n_lib = m_extent;
    ldc_b = n_stride_out;
    a_idx = in1_idx;
    b_idx = in0_idx;
    a_strides = {k_stride_in1, n_stride_in1};
    b_strides = {m_stride_in0, k_stride_in0};
    a_tensor_label = "in1";
    b_tensor_label = "in0";
    a_axes_label = "{K, N}";
    b_axes_label = "{M, K}";
  }

  // Per-operand classification. Failure means neither role axis is
  // unit-stride on the operand.
  int64_t lda_b, ldb_b;
  bool trans_a_bit, trans_b_bit;
  if (!resolve_operand(a_strides, bytes, lda_b, trans_a_bit)) {
    throw LoweringException("blas: contraction primitive '" + prim.id + "' requires one of " +
                            a_tensor_label + "'s " + a_axes_label + " role axes to be unit-stride");
  }
  if (!resolve_operand(b_strides, bytes, ldb_b, trans_b_bit)) {
    throw LoweringException("blas: contraction primitive '" + prim.id + "' requires one of " +
                            b_tensor_label + "'s " + b_axes_label + " role axes to be unit-stride");
  }
  const CBLAS_TRANSPOSE trans_a = trans_a_bit ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE trans_b = trans_b_bit ? CblasTrans : CblasNoTrans;

  if (lda_b % bytes != 0 || ldb_b % bytes != 0 || ldc_b % bytes != 0) {
    throw LoweringException("blas: leading-dimension byte strides of primitive '" + prim.id +
                            "' are not multiples of the element width " + std::to_string(bytes));
  }
  auto state = std::make_unique<BlasContractionState>();
  state->a_tensor_idx = a_idx;
  state->b_tensor_idx = b_idx;
  state->c_tensor_idx = out_idx;
  state->trans_a = trans_a;
  state->trans_b = trans_b;
  state->m_lib = m_lib;
  state->n_lib = n_lib;
  state->k_extent = k_extent;
  state->lda = lda_b / bytes;
  state->ldb = ldb_b / bytes;
  state->ldc = ldc_b / bytes;
  KernelFn kernel;
  if (dtype == "f32") {
    kernel = &blas_gemm_f32_kernel;
  } else if (dtype == "f64") {
    kernel = &blas_gemm_f64_kernel;
  } else {
    throw LoweringException("blas: contraction unsupported dtype " + dtype);
  }
  return CompiledInvocation{
      kernel,
      {state.release(), &state_deleter<BlasContractionState>},
  };
#else
  (void)m_strides_in0;
  (void)m_strides_out;
  (void)n_strides_in1;
  (void)n_strides_out;
  (void)k_strides_in0;
  (void)k_strides_in1;
  (void)m_extents;
  (void)n_extents;
  (void)k_extents;
  throw LoweringException("blas: contraction primitive '" + prim.id +
                          "' requires a BLAS implementation, which is not compiled into"
                          " this build");
#endif
}

// --------------------------------------------------------------------------
// Per-vendor sequential enforcement
// --------------------------------------------------------------------------

void enforce_sequential_blas() {
#if ETOPS_BLAS_AVAILABLE
#  if TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_OPENBLAS
  openblas_set_num_threads(1);
#  elif TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_MKL
  // Process-wide setting; the thread-local form does not propagate to
  // OpenMP workers the teir scheduler may spawn.
  mkl_set_num_threads(1);
  mkl_set_num_threads_local(1);
#  endif
  // Accelerate has no per-process or per-thread sequential knob; we accept
  // the threading model as configured by the system.
#endif
}

} // namespace

void register_blas_primitives() {
#if ETOPS_BLAS_AVAILABLE
  enforce_sequential_blas();
  register_primitive("blas", "Zero", blas_zero);
  register_primitive("blas", "Copy", blas_copy);
  register_primitive("blas", "ReLU", blas_relu);
  register_primitive("blas", "Contraction", blas_contraction);
#endif
}

} // namespace teir::blas
