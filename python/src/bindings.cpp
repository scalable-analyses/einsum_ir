#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "TensorOperation.h"

namespace py  = pybind11;
using einsum_ir::py::TensorOperation;

PYBIND11_MODULE(_etops_core, m) {
  py::enum_<TensorOperation::error_t>(m, "ErrorType")
    .value("success", TensorOperation::error_t::success)
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
      &TensorOperation::setup,
      R"doc(
        Setup for a binary tensor contraction or a unary tensor operation.
        
        :param dtype: Datatype of all tensor elements.
        :param prim_first: Type of the first touch primitive.
        :param prim_main: Type of the main primitive.
        :param prim_last: Type of the last touch primitive.
        :param dim_types: Dimension types (c, m, n, or k).
        :param exec_types: Execution types of the dimensions (prim, seq, shared, or sfc).
        :param dim_sizes: Sizes of the dimensions.
        :param strides_in0: Strides of the first input tensor.
        :param strides_in1: Strides of the second input tensor (ignored if unary).
        :param strides_out: Strides of the output tensor.
        :param num_threads: Number of threads to use for execution.
        :return: Appropiate error code.
      )doc",
      py::arg("dtype"),
      py::arg("prim_first"),
      py::arg("prim_main"),
      py::arg("prim_last"),
      py::arg("dim_types"),
      py::arg("exec_types"),
      py::arg("dim_sizes"),
      py::arg("strides_in0"),
      py::arg("strides_in1"),
      py::arg("strides_out"),
      py::arg("num_threads")
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
        
        :param in0: Pointer to the first input tensor data.
        :param in1: Pointer to the second input tensor data (ignored if unary).
        :param out: Pointer to the output tensor data.
      )doc",
      py::arg("in0"),
      py::arg("in1") = py::none(),
      py::arg("out")
    )
    .def_static(
      "optimize",
      [](
        TensorOperation::dtype_t               dtype,
        TensorOperation::prim_t                prim_first,
        TensorOperation::prim_t                prim_main,
        TensorOperation::prim_t                prim_last,
        std::vector<TensorOperation::dim_t>  & dim_types,
        std::vector<TensorOperation::exec_t> & exec_types,
        std::vector<int64_t>                 & dim_sizes,
        std::vector<int64_t>                 & strides_in0,
        std::vector<int64_t>                 & strides_in1,
        std::vector<int64_t>                 & strides_out,
        int64_t                                target_m,
        int64_t                                target_n,
        int64_t                                target_k,
        int64_t                                num_threads,
        bool                                   br_gemm_support,
        bool                                   packed_gemm_support,
        int64_t                                l2_cache_size
      ) -> py::tuple {
        // Call the static optimize function with references
        TensorOperation::error_t err = TensorOperation::optimize(
          dtype,
          prim_first,
          prim_main,
          prim_last,
          dim_types,
          exec_types,
          dim_sizes,
          strides_in0,
          strides_in1,
          strides_out,
          target_m,
          target_n,
          target_k,
          num_threads,
          br_gemm_support,
          packed_gemm_support,
          l2_cache_size
        );
        
        // Return tuple of (error, optimized_parameters)
        return py::make_tuple(
          err,
          dtype,
          prim_first,
          prim_main,
          prim_last,
          dim_types,
          exec_types,
          dim_sizes,
          strides_in0,
          strides_in1,
          strides_out
        );
      },
      R"doc(
        Optimize the tensor operation parameters.
        
        :param dtype: Datatype of all tensor elements.
        :param prim_first: Type of the first touch primitive.
        :param prim_main: Type of the main primitive.
        :param prim_last: Type of the last touch primitive.
        :param dim_types: Dimension types (c, m, n, or k).
        :param exec_types: Execution types of the dimensions (prim, seq, shared, or sfc).
        :param dim_sizes: Sizes of the dimensions.
        :param strides_in0: Strides of the first input tensor.
        :param strides_in1: Strides of the second input tensor (ignored if unary).
        :param strides_out: Strides of the output tensor.
        :param target_m: Target size for dimension m.
        :param target_n: Target size for dimension n.
        :param target_k: Target size for dimension k.
        :param num_threads: Number of threads to use for execution.
        :param br_gemm_support: Whether to support BR_GEMM optimizations.
        :param packed_gemm_support: Whether to support packed GEMM optimizations.
        :param l2_cache_size: Size of L2 cache in bytes.
        :return: Tuple containing error code and optimized parameters.
      )doc",
      py::arg("dtype"),
      py::arg("prim_first"),
      py::arg("prim_main"),
      py::arg("prim_last"),
      py::arg("dim_types"),
      py::arg("exec_types"),
      py::arg("dim_sizes"),
      py::arg("strides_in0"),
      py::arg("strides_in1"),
      py::arg("strides_out"),
      py::arg("target_m"),
      py::arg("target_n"),
      py::arg("target_k"),
      py::arg("num_threads"),
      py::arg("br_gemm_support"),
      py::arg("packed_gemm_support"),
      py::arg("l2_cache_size")
    );
}
