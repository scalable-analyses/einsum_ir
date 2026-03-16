#ifndef EINSUM_IR_PY_MODEL_H
#define EINSUM_IR_PY_MODEL_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/common.h"
#include "types.h"

namespace einsum_ir {
  namespace py {

    /// performance model type
    enum class model_t : uint32_t {
      zen5 = 0,
      m4 = 1,
      a76 = 2,
      generic = 3
    };

    /**
     * Performance prediction model for tensor operations.
     *
     * This class provides performance predictions for GEMM/BRGEMM operations.
     * The Model is constructed with just the microarchitecture, and the
     * predict() method takes the operation configuration.
     */
    class Model {
     public:
      /// execution type
      using exec_t = einsum_ir::py::exec_t;

      /// primitive type
      using prim_t = einsum_ir::py::prim_t;

      /// dimension type
      using dim_t = einsum_ir::py::dim_t;

      /// data type
      using dtype_t = einsum_ir::py::dtype_t;

      /**
       * Construct a Model with microarchitecture configuration.
       *
       * @param model_type The performance model to use (zen5, m4, a76, or generic).
       * @param peak_gflops Peak GFLOPS for generic model (required if model_type is generic).
       * @param vector_size Vector width for generic model (required if model_type is generic).
       */
      Model(model_t model_type = model_t::generic,
            double peak_gflops = 0.0,
            int vector_size = 0);

      /**
       * Predict the execution time for the tensor operation.
       *
       * @param prim_main The main primitive type (gemm or brgemm).
       * @param dim_types Dimension types for each dimension.
       * @param exec_types Execution types for each dimension.
       * @param dim_sizes Sizes of each dimension.
       * @param strides 3D stride tensor.
       * @param dtype The data type (fp32 or fp64).
       * @return Estimated execution time in seconds.
       */
      double predict(prim_t prim_main,
                     std::vector<dim_t> const& dim_types,
                     std::vector<exec_t> const& exec_types,
                     std::vector<int64_t> const& dim_sizes,
                     std::vector<std::vector<std::vector<int64_t>>> const& strides,
                     dtype_t dtype = dtype_t::fp32) const;

      /**
       * Predict the GFLOPS for a single GEMM operation.
       *
       * @param prim_main The main primitive type (gemm or brgemm).
       * @param dim_types Dimension types for each dimension.
       * @param exec_types Execution types for each dimension.
       * @param dim_sizes Sizes of each dimension.
       * @param strides 3D stride tensor.
       * @param dtype The data type (fp32 or fp64).
       * @return Estimated GFLOPS for one GEMM iteration.
       */
      double predict_gflops(prim_t prim_main,
                            std::vector<dim_t> const& dim_types,
                            std::vector<exec_t> const& exec_types,
                            std::vector<int64_t> const& dim_sizes,
                            std::vector<std::vector<std::vector<int64_t>>> const& strides,
                            dtype_t dtype = dtype_t::fp32) const;

     private:

      // Architecture configuration
      model_t m_model_type;
      double m_peak_gflops;
      int m_vector_size;
      
      /**
       * Convert model_t to common::Model.
       * @return Converted common::Model value.
       */
      einsum_ir::model::common::Model convert_model_type() const;

      /**
       * Convert dtype_t to common::DType.
       * @param dtype TensorOperation data type.
       * @return Converted common::DType value.
       */
      static einsum_ir::model::common::DType convert_dtype(dtype_t dtype);

      /**
       * Extract primitive dimensions from configuration.
       * @param prim_main Main primitive type.
       * @param dim_types Dimension types vector.
       * @param exec_types Execution types vector.
       * @param dim_sizes Dimension sizes vector.
       * @param o_m Output M dimension size.
       * @param o_n Output N dimension size.
       * @param o_k Output K dimension size.
       * @param o_br Output batch-reduce dimension size (for BRGEMM).
       */
      static void extract_primitive_dims( prim_t prim_main,
                                          std::vector<dim_t> const& dim_types,
                                          std::vector<exec_t> const& exec_types,
                                          std::vector<int64_t> const& dim_sizes,
                                          int64_t& o_m,
                                          int64_t& o_n,
                                          int64_t& o_k,
                                          int64_t& o_br);

      /**
       * Extract transpose flags from stride patterns.
       * @param dim_types Dimension types vector.
       * @param exec_types Execution types vector.
       * @param strides 3D stride tensor
       * @param o_trans_a Output transpose flag for input A.
       * @param o_trans_b Output transpose flag for input B.
       */
      static void extract_transpose_flags( std::vector<dim_t> const& dim_types,
                                           std::vector<exec_t> const& exec_types,
                                           std::vector<std::vector<std::vector<int64_t>>> const& strides,
                                           bool& o_trans_a,
                                           bool& o_trans_b);

      /**
       * Compute number of GEMM iterations.
       * @param exec_types Execution types vector.
       * @param dim_sizes Dimension sizes vector.
       * @return Number of GEMM iterations based on non-primitive dimensions.
       */
      static int64_t compute_gemm_iter( std::vector<exec_t> const& exec_types,
                                        std::vector<int64_t> const& dim_sizes);

    };

  }  // namespace py
}  // namespace einsum_ir

#endif  // EINSUM_IR_PY_MODEL_H
