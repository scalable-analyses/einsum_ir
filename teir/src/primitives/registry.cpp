#include "teir/primitive.h"

#include "internal/hash.h"
#include "teir/exception.h"

#include <string>
#include <unordered_map>

namespace teir {
namespace {

struct RegistryKey {
  std::string backend;
  std::string operation;

  bool operator==(const RegistryKey& other) const noexcept {
    return backend == other.backend && operation == other.operation;
  }
};

struct RegistryHash {
  std::size_t operator()(const RegistryKey& k) const noexcept {
    const std::size_t a = std::hash<std::string>{}(k.backend);
    const std::size_t b = std::hash<std::string>{}(k.operation);
    return internal::hash_combine(a, b);
  }
};

// Backend registration happens during static init (PYBIND11_MODULE and
// test setup) before any worker thread can call lookup_primitive, so the
// registry is effectively immutable by the time it is read.
using RegistryMap = std::unordered_map<RegistryKey, CompileFn, RegistryHash>;

RegistryMap& registry() {
  static RegistryMap r;
  return r;
}

} // namespace

void register_primitive(std::string_view backend, std::string_view operation, CompileFn fn) {
  registry()[{std::string(backend), std::string(operation)}] = fn;
}

CompileFn lookup_primitive(std::string_view backend, std::string_view operation) {
  auto& r = registry();
  auto it = r.find({std::string(backend), std::string(operation)});
  if (it == r.end()) {
    throw LoweringException("no primitive lowering registered for " + std::string(backend) +
                            "::" + std::string(operation));
  }
  return it->second;
}

bool has_primitive(std::string_view backend, std::string_view operation) {
  auto& r = registry();
  return r.contains({std::string(backend), std::string(operation)});
}

} // namespace teir
