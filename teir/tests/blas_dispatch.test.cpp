/// Tests for the BLAS backend's CBLAS dispatch.

#include "teir/config.h"
#include "teir/exception.h"
#include "teir/ir.h"
#include "teir/primitive.h"
#include "teir/runtime.h"
#include "test_common.h"

#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#if ETOPS_BLAS_AVAILABLE

namespace teir::blas {
void register_blas_primitives();
}

namespace {

using teir::tests::dispatch_atol;
using teir::tests::SEED;

template <typename T>
teir::Teir build_gemm_teir(const std::string& dtype, int64_t m, int64_t n, int64_t k) {
  const int64_t bytes = sizeof(T);
  teir::Teir t;
  t.tensors = {
      {"in0", teir::DataType{dtype, static_cast<int64_t>(bytes * 8)}},
      {"in1", teir::DataType{dtype, static_cast<int64_t>(bytes * 8)}},
      {"out", teir::DataType{dtype, static_cast<int64_t>(bytes * 8)}},
  };
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
void run_case(const std::string& dtype, int64_t m, int64_t n, int64_t k) {
  std::mt19937_64 rng(SEED);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> in0(static_cast<std::size_t>(m * k));
  std::vector<T> in1(static_cast<std::size_t>(k * n));
  for (auto& v : in0) {
    v = static_cast<T>(dist(rng));
  }
  for (auto& v : in1) {
    v = static_cast<T>(dist(rng));
  }
  std::vector<T> out_blas(static_cast<std::size_t>(m * n), T(0));
  std::vector<T> out_ref(static_cast<std::size_t>(m * n), T(0));

  teir::Teir teir = build_gemm_teir<T>(dtype, m, n, k);
  auto op = teir::compile(std::move(teir), "blas");
  std::vector<void*> bases = {in0.data(), in1.data(), out_blas.data()};
  op->execute(bases);

  reference_gemm<T>(m, n, k, in0.data(), k, in1.data(), n, out_ref.data(), n);
  const double tol = dispatch_atol<T>(k);
  for (std::size_t i = 0; i < out_blas.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(out_blas[i]),
                 Catch::Matchers::WithinAbs(static_cast<double>(out_ref[i]), tol));
  }
}

} // namespace

TEST_CASE("BLAS GEMM dispatch matches the reference loop", "[blas][dispatch]") {
  teir::blas::register_blas_primitives();
  SECTION("f32") {
    run_case<float>("f32", /*m=*/8, /*n=*/12, /*k=*/16);
  }
  SECTION("f64") {
    run_case<double>("f64", /*m=*/4, /*n=*/6, /*k=*/8);
  }
}

#endif // ETOPS_BLAS_AVAILABLE
