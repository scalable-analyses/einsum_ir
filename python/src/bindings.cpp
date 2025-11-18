#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <set>
#include "TensorOperation.h"

namespace py  = pybind11;
using einsum_ir::py::TensorOperation;

PYBIND11_MODULE(_etops_core, m) {
  py::enum_<TensorOperation::error_t>(m, "ErrorType")
    .value("success", TensorOperation::error_t::success)
    .value("compilation_failed", TensorOperation::error_t::compilation_failed)
    .value("invalid_stride_shape", TensorOperation::error_t::invalid_stride_shape)
    .value("invalid_optimization_config", TensorOperation::error_t::invalid_optimization_config)
    .export_values();

  py::enum_<TensorOperation::dtype_t>(m, "DataType" )
    .value("float32",  TensorOperation::dtype_t::fp32)
    .value("float64",  TensorOperation::dtype_t::fp64)
    .export_values();

  py::enum_<TensorOperation::prim_t>(m, "PrimType")
    .value("none",   TensorOperation::prim_t::none)
    .value("zero",   TensorOperation::prim_t::zero)
    .value("relu",   TensorOperation::prim_t::relu)
    .value("copy",   TensorOperation::prim_t::copy)
    .value("gemm",   TensorOperation::prim_t::gemm)
    .value("brgemm", TensorOperation::prim_t::brgemm)
    .export_values();

  py::enum_<TensorOperation::exec_t>(m, "ExecType")
    .value("prim",   TensorOperation::exec_t::prim)
    .value("seq",    TensorOperation::exec_t::seq)
    .value("shared", TensorOperation::exec_t::shared)
    .value("sfc",    TensorOperation::exec_t::sfc)
    .export_values();

  py::enum_<TensorOperation::dim_t>(m, "DimType")
    .value("c", TensorOperation::dim_t::c)
    .value("m", TensorOperation::dim_t::m)
    .value("n", TensorOperation::dim_t::n)
    .value("k", TensorOperation::dim_t::k)
    .export_values();

  py::class_<TensorOperation>(m, "TensorOperation")
    .def(py::init<>())
    .def(
      "setup",
      [](
        TensorOperation                                      & self,
        std::string                                    const & backend,
        TensorOperation::dtype_t                               dtype,
        TensorOperation::prim_t                                prim_first,
        TensorOperation::prim_t                                prim_main,
        TensorOperation::prim_t                                prim_last,
        std::vector<TensorOperation::dim_t>            const & dim_types,
        std::vector<TensorOperation::exec_t>           const & exec_types,
        std::vector<int64_t>                           const & dim_sizes,
        std::vector<std::vector<std::vector<int64_t>>> const & strides
      ) -> TensorOperation::error_t {
        // Validate backend
        if (backend != "tpp") {
          return TensorOperation::error_t::compilation_failed;
        }

        // Call TensorOperation setup (TPP backend)
        return self.setup(dtype, prim_first, prim_main, prim_last,
                         dim_types, exec_types, dim_sizes, strides);
      },
      R"doc(
        Setup for a unary tensor operation or a binary tensor contraction.

        The operation type is automatically determined from prim_main:
        - Unary operations: prim_main is 'copy' or 'zero'
        - Binary contractions: prim_main is 'gemm' or 'brgemm'

        Unary Operations (permutation or zero):
          - prim_main: copy or zero
          - dim_types: must be 'c' for all dimensions
          - prim_first: must be 'none'
          - prim_last: must be 'none'
          - strides: [LEVEL][2][DIMENSION] tensor (each level has 2 tensors: in, out)

        Binary Contractions (GEMM/BRGEMM):
          - prim_main: gemm or brgemm
          - dim_types: use m, n, k, c as appropriate for contraction semantics
          - prim_first: zero or none (first touch operation)
          - prim_last: relu or none (last touch operation)
          - strides: [LEVEL][3][DIMENSION] tensor (each level has 3 tensors: in0, in1, out)

        Strides 3D tensor structure [LEVEL][TENSOR][DIMENSION]:
          LEVEL: 0=primary layout, 1=packing, 2+=reserved for future
          TENSOR: 0=in0, 1=in1, 2=out (binary) or 0=in, 1=out (unary)
          DIMENSION: index corresponding to dim_types/dim_sizes

        :param backend: Backend identifier (currently only "tpp" is supported).
        :param dtype: Datatype of all tensor elements.
        :param prim_first: Type of the first touch primitive.
        :param prim_main: Type of the main primitive (determines operation type).
        :param prim_last: Type of the last touch primitive.
        :param dim_types: Dimension types provided by user.
        :param exec_types: Execution types of the dimensions (prim, seq, shared, or sfc).
        :param dim_sizes: Sizes of the dimensions.
        :param strides: 3D stride tensor [LEVEL][TENSOR][DIMENSION].
        :return: Appropriate error code.
      )doc",
      py::arg("backend"),
      py::arg("dtype"),
      py::arg("prim_first"),
      py::arg("prim_main"),
      py::arg("prim_last"),
      py::arg("dim_types"),
      py::arg("exec_types"),
      py::arg("dim_sizes"),
      py::arg("strides")
    )
    .def(
      "execute",
      [](
        TensorOperation & self,
        py::array_t<float, py::array::c_style | py::array::forcecast> in0,
        py::object                                                    in1,
        py::array_t<float, py::array::c_style | py::array::forcecast> out
      ) {
        self.execute(
          in0.data(),
          in1.is_none() ? nullptr : py::array(in1).data(),
          out.mutable_data()
        );
      },
      R"doc(
        Execute the tensor operation.

        For binary operations: provide all three tensor arguments.
        For unary operations: pass None for in1 argument.

        :param in0: First input tensor data.
        :param in1: Second input tensor data (pass None for unary operations).
        :param out: Output tensor data.
      )doc",
      py::arg("in0"),
      py::arg("in1") = py::none(),
      py::arg("out")
    )
    .def_static(
      "optimize",
      [](
        std::string                                    const & backend,
        TensorOperation::dtype_t                               dtype,
        TensorOperation::prim_t                                prim_first,
        TensorOperation::prim_t                                prim_main,
        TensorOperation::prim_t                                prim_last,
        std::vector<TensorOperation::dim_t>            const & dim_types,
        std::vector<TensorOperation::exec_t>           const & exec_types,
        std::vector<int64_t>                           const & dim_sizes,
        std::vector<std::vector<std::vector<int64_t>>> const & strides,
        py::dict                                       const & optimization_config_dict
      ) -> py::tuple {
        // Validate backend
        if( backend != "tpp") {
          std::vector<std::vector<std::vector<int64_t>>> empty_strides;
          return py::make_tuple(
            TensorOperation::error_t::compilation_failed,
            dtype, prim_first, prim_main, prim_last,
            dim_types, exec_types, dim_sizes, empty_strides
          );
        }

        // Parse the Python dict and create OptimizationConfig struct
        einsum_ir::py::OptimizationConfig l_optimization_config;

        // Get defaults first
        l_optimization_config = TensorOperation::get_default_optimization_config();

        // Valid keys for TPP backend
        std::set<std::string> valid_keys = {
          "target_m", "target_n", "target_k", "num_threads",
          "br_gemm_support", "packed_gemm_support", "packing_support",
          "sfc_support", "l2_cache_size"
        };

        try {
          // Check for unknown keys
          for (auto item : optimization_config_dict) {
            std::string key = item.first.cast<std::string>();
            if (valid_keys.find(key) == valid_keys.end()) {
              // Return error for unknown key
              std::vector<std::vector<std::vector<int64_t>>> empty_strides;
              return py::make_tuple(
                TensorOperation::error_t::invalid_optimization_config,
                dtype, prim_first, prim_main, prim_last,
                dim_types, exec_types, dim_sizes, empty_strides
              );
            }
          }

          // Override defaults with provided values
          if (optimization_config_dict.contains("target_m")) {
            l_optimization_config.target_m = optimization_config_dict["target_m"].cast<int64_t>();
          }
          if (optimization_config_dict.contains("target_n")) {
            l_optimization_config.target_n = optimization_config_dict["target_n"].cast<int64_t>();
          }
          if (optimization_config_dict.contains("target_k")) {
            l_optimization_config.target_k = optimization_config_dict["target_k"].cast<int64_t>();
          }
          if (optimization_config_dict.contains("num_threads")) {
            l_optimization_config.num_threads = optimization_config_dict["num_threads"].cast<int64_t>();
          }
          if (optimization_config_dict.contains("packed_gemm_support")) {
            l_optimization_config.packed_gemm_support = optimization_config_dict["packed_gemm_support"].cast<bool>();
          }
          if (optimization_config_dict.contains("br_gemm_support")) {
            l_optimization_config.br_gemm_support = optimization_config_dict["br_gemm_support"].cast<bool>();
          }
          if (optimization_config_dict.contains("packing_support")) {
            l_optimization_config.packing_support = optimization_config_dict["packing_support"].cast<bool>();
          }
          if (optimization_config_dict.contains("sfc_support")) {
            l_optimization_config.sfc_support = optimization_config_dict["sfc_support"].cast<bool>();
          }
          if (optimization_config_dict.contains("l2_cache_size")) {
            l_optimization_config.l2_cache_size = optimization_config_dict["l2_cache_size"].cast<int64_t>();
          }
        } catch (...) {
          // Type casting failed
          std::vector<std::vector<std::vector<int64_t>>> empty_strides;
          return py::make_tuple(
            TensorOperation::error_t::invalid_optimization_config,
            dtype, prim_first, prim_main, prim_last,
            dim_types, exec_types, dim_sizes, empty_strides
          );
        }

        // Call the static optimize function with the struct
        auto result = TensorOperation::optimize(
          dtype,
          prim_first,
          prim_main,
          prim_last,
          dim_types,
          exec_types,
          dim_sizes,
          strides,
          l_optimization_config
        );

        // Return tuple of (error, optimized_parameters)
        return py::make_tuple(
          std::get<0>(result),  // error
          std::get<1>(result),  // dtype
          std::get<2>(result),  // prim_first
          std::get<3>(result),  // prim_main
          std::get<4>(result),  // prim_last
          std::get<5>(result),  // dim_types
          std::get<6>(result),  // exec_types
          std::get<7>(result),  // dim_sizes
          std::get<8>(result)   // strides (3D)
        );
      },
      R"doc(
        Optimize the tensor operation parameters.

        The operation type is automatically determined from prim_main.
        Returns optimized configuration with potentially modified strides tensor including
        additional levels (e.g., packing) if optimization determined it beneficial.

        Binary contractions:
          Uses ContractionOptimizer with provided optimization parameters.
          May return 2 levels in strides if packing is added, otherwise 1 level.

        Unary operations:
          Uses UnaryOptimizer. Most optimization_config parameters are ignored for unary operations.
          Typically returns 1 level in strides.

        :param backend: Backend identifier (e.g., "tpp").
        :param dtype: Datatype of all tensor elements.
        :param prim_first: Type of the first touch primitive.
        :param prim_main: Type of the main primitive (determines operation type).
        :param prim_last: Type of the last touch primitive.
        :param dim_types: Dimension types.
        :param exec_types: Execution types of the dimensions (prim, seq, shared, or sfc).
        :param dim_sizes: Sizes of the dimensions.
        :param strides: 3D stride tensor [LEVEL][TENSOR][DIMENSION].
        :param optimization_config: Dictionary containing backend-specific optimization parameters.
        :return: Tuple containing (error, dtype, prim_first, prim_main, prim_last,
                 dim_types, exec_types, dim_sizes, optimized_strides).
      )doc",
      py::arg("backend"),
      py::arg("dtype"),
      py::arg("prim_first"),
      py::arg("prim_main"),
      py::arg("prim_last"),
      py::arg("dim_types"),
      py::arg("exec_types"),
      py::arg("dim_sizes"),
      py::arg("strides"),
      py::arg("optimization_config")
    )
    .def_static(
      "get_default_optimization_config",
      [](std::string const & backend) -> py::dict {
        // Validate backend
        if( backend != "tpp" ) {
          return py::dict();
        }

        // Get the struct from C++ (TPP backend)
        auto l_config = TensorOperation::get_default_optimization_config();

        // Convert to Python dict
        py::dict result;
        result["target_m"]            = l_config.target_m;
        result["target_n"]            = l_config.target_n;
        result["target_k"]            = l_config.target_k;
        result["num_threads"]         = l_config.num_threads;
        result["packed_gemm_support"] = l_config.packed_gemm_support;
        result["br_gemm_support"]     = l_config.br_gemm_support;
        result["packing_support"]     = l_config.packing_support;
        result["sfc_support"]         = l_config.sfc_support;
        result["l2_cache_size"]       = l_config.l2_cache_size;

        return result;
      },
      R"doc(
        Get default optimization configuration for a backend.

        :param backend: Backend identifier (currently only "tpp" is supported).
        :return: Dictionary containing default optimization parameters for the backend.
      )doc",
      py::arg("backend")
    );
}
