/// Python dict → `teir::Teir` conversion.
///
/// The Python side ships the IR as a nested dict of primitives types
/// (str / int / list / tuple / dict). This file converts that dict into the
/// C++ mirror types declared in `teir/include/teir/ir.h`.

#include "teir/exception.h"
#include "teir/ir.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace teir::bindings {

namespace {

DataType convert_dtype(const py::dict& d) {
  DataType dt;
  dt.name = py::cast<std::string>(d["name"]);
  dt.bits = py::cast<int64_t>(d["bits"]);
  return dt;
}

Tensor convert_tensor(const py::dict& d) {
  Tensor t;
  t.id = py::cast<std::string>(d["id"]);
  t.dtype = convert_dtype(d["dtype"]);
  return t;
}

Axis convert_axis(const py::dict& d,
                  const std::unordered_map<std::string, std::size_t>& tensor_index) {
  Axis a;
  a.id = py::cast<std::string>(d["id"]);
  a.extent = py::cast<int64_t>(d["extent"]);
  a.strides_by_tensor.assign(tensor_index.size(), 0);
  a.offsets_by_tensor.assign(tensor_index.size(), 0);
  for (const auto& [k, v] : py::cast<py::dict>(d["strides"])) {
    const std::string tid = py::cast<std::string>(k);
    auto it = tensor_index.find(tid);
    if (it == tensor_index.end()) {
      throw ValidationException("axis '" + a.id + "' references unknown tensor '" + tid +
                                "' in strides");
    }
    a.strides_by_tensor[it->second] = py::cast<int64_t>(v);
  }
  for (const auto& [k, v] : py::cast<py::dict>(d["offsets"])) {
    const std::string tid = py::cast<std::string>(k);
    auto it = tensor_index.find(tid);
    if (it == tensor_index.end()) {
      throw ValidationException("axis '" + a.id + "' references unknown tensor '" + tid +
                                "' in offsets");
    }
    a.offsets_by_tensor[it->second] = py::cast<int64_t>(v);
  }
  return a;
}

Primitive convert_primitive(const py::dict& d) {
  Primitive p;
  p.id = py::cast<std::string>(d["id"]);
  p.operation = py::cast<std::string>(d["operation"]);
  for (const auto& [k, v] : py::cast<py::dict>(d["axes"])) {
    std::string role = py::cast<std::string>(k);
    std::vector<AxisId> axes;
    for (const auto& a : py::cast<py::sequence>(v)) {
      axes.push_back(py::cast<std::string>(a));
    }
    p.axes_by_role.emplace_back(std::move(role), std::move(axes));
  }
  for (const auto& [k, v] : py::cast<py::dict>(d["metadata"])) {
    std::string key = py::cast<std::string>(k);
    p.metadata.emplace_back(std::move(key), py::cast<std::string>(v));
  }
  return p;
}

std::optional<Guard> convert_guard(const py::object& obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  Guard guard;
  for (const auto& term : py::cast<py::sequence>(obj)) {
    py::dict t = py::cast<py::dict>(term);
    const auto kind = py::cast<std::string>(t["kind"]);
    GuardTerm::Kind k;
    if (kind == "first") {
      k = GuardTerm::Kind::FIRST;
    } else if (kind == "last") {
      k = GuardTerm::Kind::LAST;
    } else {
      throw ValidationException("guard term has unknown kind '" + kind +
                                "'; expected 'first' or 'last'");
    }
    guard.push_back(GuardTerm{k, py::cast<std::string>(t["node"])});
  }
  return guard;
}

py::object guard_field(const py::dict& d) {
  if (d.contains("guard")) {
    return d["guard"];
  }
  return py::none();
}

IterationNode convert_iteration(const py::dict& d) {
  IterationNode it;
  it.id = py::cast<std::string>(d["id"]);
  it.axis = py::cast<std::string>(d["axis"]);
  std::string policy = py::cast<std::string>(d["policy"]);
  if (policy == "sequential") {
    it.policy = Policy::SEQUENTIAL;
  } else if (policy == "parallel") {
    it.policy = Policy::PARALLEL;
  } else {
    throw ValidationException("iteration node " + it.id + " has unknown policy '" + policy +
                              "'; expected 'sequential' or 'parallel'");
  }
  for (const auto& c : py::cast<py::sequence>(d["children"])) {
    it.children.push_back(py::cast<std::string>(c));
  }
  it.guard = convert_guard(guard_field(d));
  if (d.contains("num_threads")) {
    it.num_threads = py::cast<int64_t>(d["num_threads"]);
  }
  return it;
}

InvocationNode convert_invocation(const py::dict& d) {
  InvocationNode inv;
  inv.id = py::cast<std::string>(d["id"]);
  inv.primitive = py::cast<std::string>(d["primitive"]);
  inv.guard = convert_guard(guard_field(d));
  return inv;
}

} // namespace

Teir convert_teir(const py::dict& d) {
  Teir teir;
  for (const auto& t : py::cast<py::sequence>(d["tensors"])) {
    teir.tensors.push_back(convert_tensor(py::cast<py::dict>(t)));
  }
  std::unordered_map<std::string, std::size_t> tensor_index;
  tensor_index.reserve(teir.tensors.size());
  for (std::size_t i = 0; i < teir.tensors.size(); ++i) {
    tensor_index.emplace(teir.tensors[i].id, i);
  }
  for (const auto& a : py::cast<py::sequence>(d["axes"])) {
    teir.axes.push_back(convert_axis(py::cast<py::dict>(a), tensor_index));
  }
  for (const auto& p : py::cast<py::sequence>(d["primitives"])) {
    teir.primitives.push_back(convert_primitive(py::cast<py::dict>(p)));
  }
  py::dict sched = py::cast<py::dict>(d["schedule"]);
  for (const auto& r : py::cast<py::sequence>(sched["roots"])) {
    teir.schedule.roots.push_back(py::cast<std::string>(r));
  }
  for (const auto& it : py::cast<py::sequence>(sched["iterations"])) {
    teir.schedule.iterations.push_back(convert_iteration(py::cast<py::dict>(it)));
  }
  for (const auto& inv : py::cast<py::sequence>(sched["invocations"])) {
    teir.schedule.invocations.push_back(convert_invocation(py::cast<py::dict>(inv)));
  }
  return teir;
}

} // namespace teir::bindings
