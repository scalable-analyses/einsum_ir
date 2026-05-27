/// Tests for the C++ schedule walker.

#include "teir/exception.h"
#include "teir/ir.h"
#include "teir/primitive.h"
#include "teir/runtime.h"

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

/// Compact fluent builder for `teir::Teir` instances used by tests.
class TeirBuilder {
 public:
  TeirBuilder& tensor(const std::string& id) {
    t_.tensors.push_back({id, teir::DataType{"f32", 32}});
    return *this;
  }
  TeirBuilder& axis(const std::string& id, int64_t extent, std::initializer_list<int64_t> strides) {
    teir::Axis a;
    a.id = id;
    a.extent = extent;
    a.strides_by_tensor.assign(strides.begin(), strides.end());
    t_.axes.push_back(std::move(a));
    return *this;
  }
  TeirBuilder& primitive(const std::string& id, const std::string& operation) {
    teir::Primitive p;
    p.id = id;
    p.operation = operation;
    // M/N roles are schema-required; walker tests don't bind axes to them.
    p.axes_by_role.push_back({"M", {}});
    p.axes_by_role.push_back({"N", {}});
    p.metadata = {{"data_type", "f32"}};
    t_.primitives.push_back(std::move(p));
    return *this;
  }
  TeirBuilder& iter(const std::string& id,
                    const std::string& axis,
                    std::initializer_list<std::string> children,
                    teir::Policy policy = teir::Policy::SEQUENTIAL) {
    teir::IterationNode it;
    it.id = id;
    it.axis = axis;
    it.policy = policy;
    for (const auto& c : children) {
      it.children.push_back(c);
    }
    t_.schedule.iterations.push_back(std::move(it));
    return *this;
  }
  TeirBuilder& invoke(const std::string& id,
                      const std::string& primitive,
                      std::optional<teir::Guard> guard = std::nullopt) {
    teir::InvocationNode inv;
    inv.id = id;
    inv.primitive = primitive;
    inv.guard = std::move(guard);
    t_.schedule.invocations.push_back(std::move(inv));
    return *this;
  }
  TeirBuilder& roots(std::initializer_list<std::string> ids) {
    for (const auto& r : ids) {
      t_.schedule.roots.push_back(r);
    }
    return *this;
  }
  teir::Teir build() { return std::move(t_); }

 private:
  teir::Teir t_;
};

teir::Guard guard_first(const std::string& node) {
  return teir::Guard{teir::GuardTerm{teir::GuardTerm::Kind::FIRST, node}};
}

teir::Teir scalar_copy_teir() {
  return TeirBuilder()
      .tensor("in0")
      .tensor("out")
      .axis("a", 4, {4, 4})
      .primitive("copy", "Copy")
      .iter("iter", "a", {"inv"})
      .invoke("inv", "copy")
      .roots({"iter"})
      .build();
}

// Walker-test counters. The registry only accepts raw function pointers,
// so test primitives are free functions that route through a shared atomic
// counter (one per test case below).

std::atomic<int64_t> g_walker_count{0};
void ut_walker_copy_kernel(const teir::TileAddresses& tiles, const void*) {
  *static_cast<float*>(tiles.addrs[1]) = *static_cast<const float*>(tiles.addrs[0]);
  g_walker_count.fetch_add(1, std::memory_order_relaxed);
}

std::atomic<int64_t> g_guard_count{0};
void ut_guard_zero_kernel(const teir::TileAddresses&, const void*) {
  g_guard_count.fetch_add(1, std::memory_order_relaxed);
}

void ut_noop_kernel(const teir::TileAddresses&, const void*) {}

std::atomic<int64_t> g_forest_count{0};
void ut_forest_zero_kernel(const teir::TileAddresses&, const void*) {
  g_forest_count.fetch_add(1, std::memory_order_relaxed);
}

std::atomic<int64_t> g_collapse_count{0};
void ut_collapse_set_kernel(const teir::TileAddresses& tiles, const void*) {
  *static_cast<float*>(tiles.addrs[0]) = 1.0f;
  g_collapse_count.fetch_add(1, std::memory_order_relaxed);
}

teir::CompiledInvocation make_stateless(teir::KernelFn k) {
  return teir::CompiledInvocation{k, {nullptr, [](void*) {}}};
}

teir::CompiledInvocation ut_walker_copy(const teir::Primitive&, const teir::Teir&) {
  return make_stateless(&ut_walker_copy_kernel);
}
teir::CompiledInvocation ut_guard_zero(const teir::Primitive&, const teir::Teir&) {
  return make_stateless(&ut_guard_zero_kernel);
}
teir::CompiledInvocation ut_noop(const teir::Primitive&, const teir::Teir&) {
  return make_stateless(&ut_noop_kernel);
}
teir::CompiledInvocation ut_forest_zero(const teir::Primitive&, const teir::Teir&) {
  return make_stateless(&ut_forest_zero_kernel);
}
teir::CompiledInvocation ut_collapse_set(const teir::Primitive&, const teir::Teir&) {
  return make_stateless(&ut_collapse_set_kernel);
}

} // namespace

TEST_CASE("Operation::execute walks the schedule and dispatches the primitive", "[runtime]") {
  g_walker_count.store(0);
  teir::register_primitive("ut_walker", "Copy", &ut_walker_copy);

  std::vector<float> in_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out_data(4, 0.0f);
  auto op = teir::compile(scalar_copy_teir(), "ut_walker");
  std::vector<void*> bases = {in_data.data(), out_data.data()};
  op->execute(bases);
  REQUIRE(g_walker_count.load() == 4);
  REQUIRE(out_data == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
}

TEST_CASE("Operation::execute respects first/last guards", "[runtime]") {
  g_guard_count.store(0);
  teir::register_primitive("ut_guard", "Zero", &ut_guard_zero);

  auto t = TeirBuilder()
               .tensor("out")
               .axis("a", 3, {1})
               .axis("b", 5, {1})
               .primitive("zero", "Zero")
               .invoke("inv", "zero", guard_first("iter_a"))
               .iter("iter_b", "b", {"inv"})
               .iter("iter_a", "a", {"iter_b"})
               .roots({"iter_a"})
               .build();

  std::vector<float> out(64, 0.0f);
  auto op = teir::compile(std::move(t), "ut_guard");
  std::vector<void*> bases = {out.data()};
  op->execute(bases);
  REQUIRE(g_guard_count.load() == 5);
}

TEST_CASE("Operation::compile rejects missing primitive lowerings", "[runtime]") {
  REQUIRE_THROWS_AS(teir::compile(scalar_copy_teir(), "ut_does_not_exist"),
                    teir::LoweringException);
}

TEST_CASE("Operation::execute raises ValidationException for unscoped guard nodes",
          "[runtime][guards]") {
  teir::register_primitive("ut_guard_scope", "Zero", &ut_noop);

  auto t = TeirBuilder()
               .tensor("out")
               .axis("a", 2, {1})
               .axis("b", 2, {1})
               .primitive("zero", "Zero")
               .invoke("inv_a", "zero")
               .invoke("inv_b", "zero", guard_first("iter_a"))
               .iter("iter_a", "a", {"inv_a"})
               .iter("iter_b", "b", {"inv_b"})
               .roots({"iter_a", "iter_b"})
               .build();

  REQUIRE_THROWS_AS(teir::compile(std::move(t), "ut_guard_scope"), teir::ValidationException);
}

TEST_CASE("Operation::execute raises ValidationException for unknown guard nodes",
          "[runtime][guards]") {
  teir::register_primitive("ut_guard_unknown", "Zero", &ut_noop);

  auto t = TeirBuilder()
               .tensor("out")
               .axis("a", 2, {1})
               .primitive("zero", "Zero")
               .invoke("inv", "zero", guard_first("does_not_exist"))
               .iter("iter_a", "a", {"inv"})
               .roots({"iter_a"})
               .build();

  REQUIRE_THROWS_AS(teir::compile(std::move(t), "ut_guard_unknown"), teir::ValidationException);
}

TEST_CASE("Operation::execute raises ValidationException for guard targeting invocation",
          "[runtime][guards]") {
  teir::register_primitive("ut_guard_inv_target", "Zero", &ut_noop);

  auto t = TeirBuilder()
               .tensor("out")
               .axis("a", 2, {1})
               .primitive("zero", "Zero")
               .invoke("inv_sibling", "zero")
               .invoke("inv", "zero", guard_first("inv_sibling"))
               .iter("iter_a", "a", {"inv_sibling", "inv"})
               .roots({"iter_a"})
               .build();

  REQUIRE_THROWS_AS(teir::compile(std::move(t), "ut_guard_inv_target"), teir::ValidationException);
}

TEST_CASE("Operation::execute walks parallel iteration with a multi-root forest",
          "[runtime][parallel][forest]") {
  g_forest_count.store(0);
  teir::register_primitive("ut_par_forest", "Zero", &ut_forest_zero);

  auto t = TeirBuilder()
               .tensor("out")
               .axis("a", 8, {4})
               .axis("b", 4, {4})
               .primitive("inc", "Zero")
               .invoke("inv_seq", "inc")
               .invoke("inv_par", "inc")
               .iter("root_seq", "a", {"inv_seq"})
               .iter("root_par", "b", {"inv_par"}, teir::Policy::PARALLEL)
               .roots({"root_seq", "root_par"})
               .build();

  std::vector<float> out(32, 0.0f);
  auto op = teir::compile(std::move(t), "ut_par_forest");
  std::vector<void*> bases = {out.data()};
  op->execute(bases);
  REQUIRE(g_forest_count.load() == 12);
}

TEST_CASE("Operation::execute collapses a chain of parallel iteration nodes",
          "[runtime][parallel][collapse]") {
  g_collapse_count.store(0);
  teir::register_primitive("ut_collapse", "Zero", &ut_collapse_set);

  // Two nested parallel iteration nodes over axes a (extent 3) and b
  // (extent 4) form a chain that the walker should iterate as one
  // collapsed parallel domain of 12 work items. The output strides give
  // every (i_a, i_b) pair a distinct float slot; if the decode is wrong,
  // collisions or skipped slots will leave non-1.0 values behind.
  auto t = TeirBuilder()
               .tensor("out")
               .axis("a", 3, {16})
               .axis("b", 4, {4})
               .primitive("zero", "Zero")
               .invoke("inv", "zero")
               .iter("iter_b", "b", {"inv"}, teir::Policy::PARALLEL)
               .iter("iter_a", "a", {"iter_b"}, teir::Policy::PARALLEL)
               .roots({"iter_a"})
               .build();

  std::vector<float> out(12, 0.0f);
  auto op = teir::compile(std::move(t), "ut_collapse");
  std::vector<void*> bases = {out.data()};
  op->execute(bases);
  REQUIRE(g_collapse_count.load() == 12);
  for (float v : out) {
    REQUIRE(v == 1.0f);
  }
}
