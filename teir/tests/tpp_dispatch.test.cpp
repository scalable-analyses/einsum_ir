/// Tests for the TPP backend's libxsmm-JIT dispatch path.
///
/// The dispatched kernel must produce numerically close results to a
/// portable scalar accumulator. FP addition is non-associative.
/// libxsmm may use a different reduction order from the reference loop.

#include "teir/config.h"
#include "teir/exception.h"
#include "teir/ir.h"
#include "teir/primitive.h"
#include "teir/runtime.h"
#include "test_common.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace {

using teir::tests::dispatch_atol;
using teir::tests::SEED;

template <typename T>
T draw(std::mt19937_64& rng) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  return static_cast<T>(dist(rng));
}

template <typename T>
std::vector<T> seed_buffer(std::size_t n, std::mt19937_64& rng) {
  std::vector<T> buf(n);
  for (auto& v : buf) {
    v = draw<T>(rng);
  }
  return buf;
}

template <typename T>
void reference_gemm(int64_t m,
                    int64_t n,
                    int64_t k,
                    const T* a,
                    int64_t lda,
                    const T* b,
                    int64_t ldb,
                    T* c,
                    int64_t ldc) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      T acc = c[i * ldc + j];
      for (int64_t p = 0; p < k; ++p) {
        acc += a[i * lda + p] * b[p * ldb + j];
      }
      c[i * ldc + j] = acc;
    }
  }
}

template <typename T>
void reference_brgemm(int64_t batch,
                      int64_t m,
                      int64_t n,
                      int64_t k,
                      const T* a_base,
                      int64_t lda,
                      int64_t a_stride_elements,
                      const T* b_base,
                      int64_t ldb,
                      int64_t b_stride_elements,
                      T* c,
                      int64_t ldc) {
  for (int64_t r = 0; r < batch; ++r) {
    reference_gemm<T>(
        m, n, k, a_base + r * a_stride_elements, lda, b_base + r * b_stride_elements, ldb, c, ldc);
  }
}

template <typename T>
teir::Teir build_gemm_teir(const std::string& dtype, int64_t m, int64_t n, int64_t k) {
  const int64_t bytes = sizeof(T);
  teir::Teir t;
  t.tensors = {
      {.id = "in0",
       .dtype = teir::DataType{.name = dtype, .bits = static_cast<int64_t>(bytes * 8)}},
      {.id = "in1",
       .dtype = teir::DataType{.name = dtype, .bits = static_cast<int64_t>(bytes * 8)}},
      {.id = "out",
       .dtype = teir::DataType{.name = dtype, .bits = static_cast<int64_t>(bytes * 8)}},
  };
  // in0 has shape (m, k) row-major; in1 has shape (k, n) row-major;
  // out has shape (m, n) row-major.
  teir::Axis m_axis;
  m_axis.id = "m";
  m_axis.extent = m;
  m_axis.strides_by_tensor = {k * bytes, 0, n * bytes};
  teir::Axis n_axis;
  n_axis.id = "n";
  n_axis.extent = n;
  n_axis.strides_by_tensor = {0, bytes, bytes};
  teir::Axis k_axis;
  k_axis.id = "k";
  k_axis.extent = k;
  k_axis.strides_by_tensor = {bytes, n * bytes, 0};
  t.axes = {m_axis, n_axis, k_axis};

  teir::Primitive p;
  p.id = "gemm";
  p.operation = "Contraction";
  p.axes_by_role = {{"M", {"m"}}, {"N", {"n"}}, {"K", {"k"}}};
  p.metadata = {{"data_type", dtype}};
  t.primitives = {p};

  teir::InvocationNode inv;
  inv.id = "inv";
  inv.primitive = "gemm";
  t.schedule.roots = {"inv"};
  t.schedule.invocations = {inv};
  return t;
}

template <typename T>
teir::Teir
build_brgemm_teir(const std::string& dtype, int64_t batch, int64_t m, int64_t n, int64_t k) {
  const int64_t bytes = sizeof(T);
  teir::Teir t;
  t.tensors = {
      {.id = "in0",
       .dtype = teir::DataType{.name = dtype, .bits = static_cast<int64_t>(bytes * 8)}},
      {.id = "in1",
       .dtype = teir::DataType{.name = dtype, .bits = static_cast<int64_t>(bytes * 8)}},
      {.id = "out",
       .dtype = teir::DataType{.name = dtype, .bits = static_cast<int64_t>(bytes * 8)}},
  };
  // in0 stacked: shape (batch, m, k); in1 stacked: shape (batch, k, n);
  // out shape (m, n).
  teir::Axis batch_axis;
  batch_axis.id = "b";
  batch_axis.extent = batch;
  batch_axis.strides_by_tensor = {m * k * bytes, k * n * bytes, 0};
  teir::Axis m_axis;
  m_axis.id = "m";
  m_axis.extent = m;
  m_axis.strides_by_tensor = {k * bytes, 0, n * bytes};
  teir::Axis n_axis;
  n_axis.id = "n";
  n_axis.extent = n;
  n_axis.strides_by_tensor = {0, bytes, bytes};
  teir::Axis k_axis;
  k_axis.id = "k";
  k_axis.extent = k;
  k_axis.strides_by_tensor = {bytes, n * bytes, 0};
  t.axes = {batch_axis, m_axis, n_axis, k_axis};

  teir::Primitive p;
  p.id = "brgemm";
  p.operation = "Contraction";
  p.axes_by_role = {{"M", {"m"}}, {"N", {"n"}}, {"K", {"b", "k"}}};
  p.metadata = {{"data_type", dtype}};
  t.primitives = {p};

  teir::InvocationNode inv;
  inv.id = "inv";
  inv.primitive = "brgemm";
  t.schedule.roots = {"inv"};
  t.schedule.invocations = {inv};
  return t;
}

template <typename T>
void run_gemm_case(const std::string& dtype, int64_t m, int64_t n, int64_t k) {
  std::mt19937_64 rng(SEED);
  auto in0 = seed_buffer<T>(static_cast<std::size_t>(m * k), rng);
  auto in1 = seed_buffer<T>(static_cast<std::size_t>(k * n), rng);
  std::vector<T> out_dispatch(static_cast<std::size_t>(m * n), T(0));
  std::vector<T> out_reference(static_cast<std::size_t>(m * n), T(0));

  teir::Teir teir = build_gemm_teir<T>(dtype, m, n, k);
  auto op = teir::compile(std::move(teir), "tpp");
  std::vector<void*> bases = {in0.data(), in1.data(), out_dispatch.data()};
  op->execute(bases);

  reference_gemm<T>(m, n, k, in0.data(), k, in1.data(), n, out_reference.data(), n);
  const double tol = dispatch_atol<T>(k);
  for (std::size_t i = 0; i < out_dispatch.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(out_dispatch[i]),
                 Catch::Matchers::WithinAbs(static_cast<double>(out_reference[i]), tol));
  }
}

template <typename T>
void run_brgemm_case(const std::string& dtype, int64_t batch, int64_t m, int64_t n, int64_t k) {
  std::mt19937_64 rng(SEED);
  auto in0 = seed_buffer<T>(static_cast<std::size_t>(batch * m * k), rng);
  auto in1 = seed_buffer<T>(static_cast<std::size_t>(batch * k * n), rng);
  std::vector<T> out_dispatch(static_cast<std::size_t>(m * n), T(0));
  std::vector<T> out_reference(static_cast<std::size_t>(m * n), T(0));

  teir::Teir teir = build_brgemm_teir<T>(dtype, batch, m, n, k);
  auto op = teir::compile(std::move(teir), "tpp");
  std::vector<void*> bases = {in0.data(), in1.data(), out_dispatch.data()};
  op->execute(bases);

  reference_brgemm<T>(
      batch, m, n, k, in0.data(), k, m * k, in1.data(), n, k * n, out_reference.data(), n);
  const double tol = dispatch_atol<T>(batch * k);
  for (std::size_t i = 0; i < out_dispatch.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(out_dispatch[i]),
                 Catch::Matchers::WithinAbs(static_cast<double>(out_reference[i]), tol));
  }
}

} // namespace

TEST_CASE("TPP GEMM dispatch matches the reference loop", "[tpp][dispatch]") {
  teir::tpp::register_tpp_primitives();
  SECTION("f32") {
    run_gemm_case<float>("f32", /*m=*/8, /*n=*/12, /*k=*/16);
  }
  SECTION("f64") {
    run_gemm_case<double>("f64", /*m=*/4, /*n=*/6, /*k=*/8);
  }
}

TEST_CASE("TPP BRGEMM dispatch matches the reference loop", "[tpp][dispatch]") {
  teir::tpp::register_tpp_primitives();
  SECTION("f32") {
    run_brgemm_case<float>("f32", /*batch=*/3, /*m=*/8, /*n=*/12, /*k=*/16);
  }
  SECTION("f64") {
    run_brgemm_case<double>("f64", /*batch=*/2, /*m=*/4, /*n=*/6, /*k=*/8);
  }
}
