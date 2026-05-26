/// Python binding entry point. Exposes a thin facade over the `teir::`
/// runtime so the Python package can compile and execute IR.

#include "teir/config.h"
#include "teir/exception.h"
#include "teir/ir.h"
#include "teir/primitive.h"
#include "teir/runtime.h"
#include "teir/threading.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace teir::bindings {
Teir convert_teir(const py::dict& d);
}

namespace {

const char* threading_backend_name() {
#if TEIR_THREADING_BACKEND == TEIR_THREADING_BACKEND_DISPATCH
  return "dispatch";
#elif TEIR_THREADING_BACKEND == TEIR_THREADING_BACKEND_OPENMP
  return "openmp";
#else
  return "sequential";
#endif
}

const char* blas_vendor_name() {
#if TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_OPENBLAS
  return "openblas";
#elif TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_MKL
  return "mkl";
#elif TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_ACCELERATE
  return "accelerate";
#elif TEIR_BLAS_VENDOR == TEIR_BLAS_VENDOR_GENERIC
  return "generic";
#else
  return "unavailable";
#endif
}

bool libxsmm_available() {
#if ETOPS_LIBXSMM_AVAILABLE
  return true;
#else
  return false;
#endif
}

bool blas_available() {
#if ETOPS_BLAS_AVAILABLE
  return true;
#else
  return false;
#endif
}

} // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "etops native extension (teir runtime bindings)";
  m.attr("__version__") = TEIR_VERSION_STRING;

  // Force primitive registration. Without an explicit call, the static
  // library's backend translation units could be dropped at link time.
#if ETOPS_LIBXSMM_AVAILABLE
  teir::tpp::register_tpp_primitives();
#endif
#if ETOPS_BLAS_AVAILABLE
  teir::blas::register_blas_primitives();
#endif

  // The three concrete subclasses are the entire public exception surface
  // on the Python side; pybind11 dispatches translators in reverse
  // registration order, so each subclass is matched directly.
  py::register_exception<teir::ValidationException>(m, "NativeValidationError");
  py::register_exception<teir::LoweringException>(m, "NativeLoweringError");
  py::register_exception<teir::RuntimeException>(m, "NativeRuntimeError");

  py::class_<teir::Operation>(m, "Operation")
      .def(
          "execute",
          [](teir::Operation& op, const std::vector<uintptr_t>& base_addresses) {
            std::vector<void*> bases;
            bases.reserve(base_addresses.size());
            for (auto addr : base_addresses) {
              bases.push_back(reinterpret_cast<void*>(addr));
            }
            op.execute(bases);
          },
          "Execute the operation against the provided tensor base addresses.",
          py::call_guard<py::gil_scoped_release>());

  m.def(
      "compile_teir",
      [](const py::dict& teir_dict, const std::string& backend) {
        teir::Teir t = teir::bindings::convert_teir(teir_dict);
        return teir::compile(std::move(t), backend);
      },
      py::arg("teir_dict"),
      py::arg("backend"),
      "Convert a Python TEIR dict to a C++ Teir and compile it.");

  py::module config = m.def_submodule("config", "Build-time configuration");
  config.attr("threading_backend") = threading_backend_name();
  config.attr("blas_vendor") = blas_vendor_name();
  config.attr("libxsmm_available") = libxsmm_available();
  config.attr("blas_available") = blas_available();
  config.def("num_threads_available",
             &teir::threading::num_threads_available,
             "Return the number of worker threads available to the runtime.");
  config.def("has_primitive",
             &teir::has_primitive,
             py::arg("backend"),
             py::arg("operation"),
             "Return True if a primitive lowering is registered.");
}
