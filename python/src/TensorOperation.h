#ifndef EINSUM_IR_PY_TENSOR_OPERATION_H
#define EINSUM_IR_PY_TENSOR_OPERATION_H

#include <cstdint>
#include <vector>
#include <einsum_ir/basic/binary/ContractionBackendTpp.h>
#include <einsum_ir/basic/binary/ContractionOptimizer.h>

namespace einsum_ir {
  namespace py {
    class TensorOperation;
  }
}

class einsum_ir::py::TensorOperation {
  public:
    /// execution type
    enum class exec_t : uint32_t {
      seq       = 0, 
      prim      = 1,
      shared    = 2,
      sfc       = 3,
      undefined = 99
    };

    /// primitive type
    enum class prim_t : uint32_t {
      none      =  0,
      zero      =  1,
      copy      =  2,
      relu      =  3,
      gemm      =  4,
      brgemm    =  5,
      undefined = 99
    };

    /// dimension type
    enum class dim_t : uint32_t {
      c         = 0, 
      m         = 1, 
      n         = 2, 
      k         = 3, 
      undefined = 99
    };

    /// data type
    enum class dtype_t : uint32_t {
      fp32 = 0,
      fp64 = 1
    };

    /// error codes
    enum class error_t : int32_t {
      success            = 0,
      compilation_failed = 1
    };

    einsum_ir::basic::ContractionBackendTpp m_backend;

    /**
     * Setup for a binary tensor contraction or a unary tensor operation.
     *
     * @param dtype       Datatype of all tensor elements.
     * @param prim_first  Type of the first touch primitive.
     * @param prim_main   Type of the main primitive.
     * @param prim_last   Type of the last touch primitive.
     * @param dim_types   Dimension types (c, m, n, or k).
     * @param exec_types  Execution type of the dimensions (prim, seq, shared, or sfc).
     * @param dim_sizes   Sizes of the dimensions.
     * @param strides_in0 Strides of the first input tensor.
     * @param strides_in1 Strides of the second input tensor (ignored if unary).
     * @param strides_out Strides of the output tensor.
     * @param num_threads_omp  Number of threads to use for normal parallelization.
     * @param num_threads_sfc_m Number of threads to use for SFC parallelization in M dimension.
     * @param num_threads_sfc_n Number of threads to use for SFC parallelization in N dimension.
     * @return            Appropiate error code.
     **/
    error_t setup(
      dtype_t                        dtype,
      prim_t                         prim_first,
      prim_t                         prim_main,
      prim_t                         prim_last,
      std::vector< dim_t >   const & dim_types,
      std::vector< exec_t >  const & exec_types,
      std::vector< int64_t > const & dim_sizes,
      std::vector< int64_t > const & strides_in0,
      std::vector< int64_t > const & strides_in1,
      std::vector< int64_t > const & strides_out,
      int64_t                       num_threads_omp,
      int64_t                       num_threads_sfc_m,
      int64_t                       num_threads_sfc_n
    );

    /**
     * Execute the tensor operation.
     *
     * @param tensor_in0 First input tensor.
     * @param tensor_in1 Second input tensor (use nullptr if unary).
     * @param tensor_out Output tensor.
     **/
    void execute( void const * tensor_in0,
                  void const * tensor_in1,
                  void       * tensor_out );

    /**
     * Optimizes a tensor operation configuration using ContractionOptimizer.
     *
     * @param dtype               Datatype of all tensor elements.
     * @param prim_first          Type of the first touch primitive.
     * @param prim_main           Type of the main primitive (modified by optimization).
     * @param prim_last           Type of the last touch primitive.
     * @param dim_types           Dimension type (modified by optimization).
     * @param exec_types          Execution type of the dimensions (modified by optimization).
     * @param dim_sizes           Sizes of the dimensions (modified by optimization).
     * @param strides_in0         Strides of the first input tensor (modified by optimization).
     * @param strides_in1         Strides of the second input tensor (modified by optimization).
     * @param strides_out         Strides of the output tensor (modified by optimization).
     * @param target_m            Target M block size for optimization.
     * @param target_n            Target N block size for optimization.
     * @param target_k            Target K block size for optimization.
     * @param num_threads         Number of threads for parallel execution (determined automatically if <1).
     * @param br_gemm_support     Whether backend supports batch-reduce GEMM.
     * @param packed_gemm_support Whether backend supports packed GEMM.
     * @param l2_cache_size       Size of L2 cache in bytes.
     * @return                    Appropriate error code.
     **/
    static error_t optimize(
      dtype_t                  dtype,
      prim_t                 & prim_first,
      prim_t                 & prim_main,
      prim_t                 & prim_last,
      std::vector< dim_t >   & dim_types,
      std::vector< exec_t >  & exec_types,
      std::vector< int64_t > & dim_sizes,
      std::vector< int64_t > & strides_in0,
      std::vector< int64_t > & strides_in1,
      std::vector< int64_t > & strides_out,
      int64_t                  target_m,
      int64_t                  target_n,
      int64_t                  target_k,
      int64_t                & num_threads_omp,
      int64_t                & num_threads_sfc_m,
      int64_t                & num_threads_sfc_n,
      bool                     generate_sfc,
      bool                     br_gemm_support,
      bool                     packed_gemm_support,
      int64_t                  l2_cache_size
    );

  private:
    /**
     * Helper function to convert TensorOperation execution types to backend execution types.
     *
     * @param exec_type TensorOperation execution type.
     * @return          Backend execution type.
     **/
    static inline einsum_ir::basic::exec_t convert_exec_type( exec_t exec_type );

    /**
     * Helper function to convert backend execution types back to TensorOperation execution types.
     *
     * @param exec_type Backend execution type.
     * @return          TensorOperation execution type.
     **/
    static inline exec_t convert_exec_type_back( einsum_ir::basic::exec_t exec_type );

    /**
     * Helper function to convert TensorOperation dimension types to backend dimension types.
     *
     * @param dim_type TensorOperation dimension type.
     * @return         Backend dimension type.
     **/
    static inline einsum_ir::basic::dim_t convert_dim_type( dim_t dim_type );

    /**
     * Helper function to convert backend dimension types back to TensorOperation dimension types.
     *
     * @param dim_type Backend dimension type.
     * @return         TensorOperation dimension type.
     **/
    static inline dim_t convert_dim_type_back( einsum_ir::basic::dim_t dim_type );

    /**
     * Helper function to convert primitive types to backend kernel types.
     *
     * @param prim_type TensorOperation primitive type.
     * @return          Backend kernel type.
     **/
    static inline einsum_ir::basic::kernel_t convert_prim_to_kernel( prim_t prim_type );

    /**
     * Helper function to create iter_property vector from input parameters.
     *
     * @param dim_types   Dimension types vector.
     * @param exec_types  Execution types vector.
     * @param dim_sizes   Dimension sizes vector.
     * @param strides_in0 Strides for first input tensor.
     * @param strides_in1 Strides for second input tensor.
     * @param strides_out Strides for output tensor.
     * @return            Vector of iteration properties.
     **/
    static inline std::vector<einsum_ir::basic::iter_property> create_iter_properties(
      std::vector<dim_t>   const & dim_types,
      std::vector<exec_t>  const & exec_types,
      std::vector<int64_t> const & dim_sizes,
      std::vector<int64_t> const & strides_in0,
      std::vector<int64_t> const & strides_in1,
      std::vector<int64_t> const & strides_out
    );

    /**
     * Helper function to update parameters from optimized iter_property vector.
     *
     * @param iters       Optimized iteration properties vector.
     * @param dim_types   Output dimension types vector.
     * @param exec_types  Output execution types vector.
     * @param dim_sizes   Output dimension sizes vector.
     * @param strides_in0 Output strides for first input tensor.
     * @param strides_in1 Output strides for second input tensor.
     * @param strides_out Output strides for output tensor.
     **/
    static inline void update_parameters_from_iters(
      std::vector<einsum_ir::basic::iter_property> const & iters,
      std::vector<dim_t>                                  & dim_types,
      std::vector<exec_t>                                 & exec_types,
      std::vector<int64_t>                                & dim_sizes,
      std::vector<int64_t>                                & strides_in0,
      std::vector<int64_t>                                & strides_in1,
      std::vector<int64_t>                                & strides_out
    );

    /**
     * Converts TensorOperation dtype to number of bytes.
     *
     * @param dtype TensorOperation data type.
     * @return      Number of bytes for the given data type.
     **/
    static inline int64_t dtype_to_num_bytes( dtype_t dtype );
};

#endif
