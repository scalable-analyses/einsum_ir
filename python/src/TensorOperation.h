#ifndef EINSUM_IR_PY_TENSOR_OPERATION_H
#define EINSUM_IR_PY_TENSOR_OPERATION_H

#include <cstdint>
#include <vector>
#include <einsum_ir/basic/unary/UnaryBackendTpp.h>
#include <einsum_ir/basic/unary/UnaryOptimizer.h>
#include <einsum_ir/basic/binary/ContractionBackendTpp.h>
#include <einsum_ir/basic/binary/ContractionOptimizer.h>

namespace einsum_ir {
  namespace py {
    class TensorOperation;

    /**
     * Backend-specific optimization parameters.
     */
    struct OptimizationConfig {
      int64_t target_m            = 0;
      int64_t target_n            = 0;
      int64_t target_k            = 0;
      int64_t num_threads         = 0;
      bool    br_gemm_support     = false;
      bool    packed_gemm_support = false;
      bool    packing_support     = false;
      bool    sfc_support         = false;
      int64_t l2_cache_size       = 0;
    };
  }
}

class einsum_ir::py::TensorOperation {
  public:
    /// operation type
    enum class op_type_t : uint32_t {
      binary    = 0,
      unary     = 1,
      undefined = 99
    };

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
      success                     = 0,
      compilation_failed          = 1,
      invalid_stride_shape        = 2,
      invalid_optimization_config = 3
    };

    op_type_t m_op_type = op_type_t::undefined;
    einsum_ir::basic::UnaryBackendTpp m_backend_unary;
    einsum_ir::basic::ContractionBackendTpp m_backend_binary;

    /**
     * Setup for a binary tensor contraction or a unary tensor operation.
     *
     * @param dtype      Datatype of all tensor elements.
     * @param prim_first Type of the first touch primitive.
     * @param prim_main  Type of the main primitive (determines operation type).
     * @param prim_last  Type of the last touch primitive.
     * @param dim_types  Dimension types.
     * @param exec_types Execution type of the dimensions (prim, seq, shared, or sfc).
     * @param dim_sizes  Sizes of the dimensions.
     * @param strides    3D stride tensor: [LEVEL][TENSOR][DIMENSION]
     *                   - LEVEL: 0=primary layout, 1=packing, 2+=reserved
     *                   - TENSOR: 0=in0, 1=in1, 2=out (binary) or 0=in, 1=out (unary)
     *                   - DIMENSION: dimension index
     * @return           Appropriate error code.
     **/
    error_t setup(
      dtype_t                                                      dtype,
      prim_t                                                       prim_first,
      prim_t                                                       prim_main,
      prim_t                                                       prim_last,
      std::vector< dim_t >                                 const & dim_types,
      std::vector< exec_t >                                const & exec_types,
      std::vector< int64_t >                               const & dim_sizes,
      std::vector< std::vector< std::vector< int64_t > > > const & strides
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
     * Optimizes a tensor operation configuration.
     *
     * The operation type is automatically determined from prim_main.
     * For binary operations, may add packing levels if optimization determines it beneficial.
     * For unary operations, returns single level.
     *
     * @param dtype               Datatype of all tensor elements.
     * @param prim_first          Type of the first touch primitive.
     * @param prim_main           Type of the main primitive.
     * @param prim_last           Type of the last touch primitive.
     * @param dim_types           Dimension types.
     * @param exec_types          Execution type of the dimensions.
     * @param dim_sizes           Sizes of the dimensions.
     * @param strides             3D stride tensor: [LEVEL][TENSOR][DIMENSION]
     * @param optimization_config Backend-specific optimization parameters.
     * @return                    Tuple of (error, dtype, prim_first, prim_main, prim_last,
     *                            dim_types, exec_types, dim_sizes, optimized_strides).
     **/
    static std::tuple<
      error_t,
      dtype_t,
      prim_t,
      prim_t,
      prim_t,
      std::vector< dim_t >,
      std::vector< exec_t >,
      std::vector< int64_t >,
      std::vector< std::vector< std::vector< int64_t > > >
    > optimize(
      dtype_t                                                      dtype,
      prim_t                                                       prim_first,
      prim_t                                                       prim_main,
      prim_t                                                       prim_last,
      std::vector< dim_t >                                 const & dim_types,
      std::vector< exec_t >                                const & exec_types,
      std::vector< int64_t >                               const & dim_sizes,
      std::vector< std::vector< std::vector< int64_t > > > const & strides,
      OptimizationConfig                                   const & optimization_config
    );

    /**
     * Get default optimization configuration.
     *
     * @return Default optimization parameters for the TPP backend.
     **/
    static OptimizationConfig get_default_optimization_config();

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
     * @param packing_in0 Output packing strides for first input tensor.
     * @param packing_in1 Output packing strides for second input tensor.
     **/
    static inline void update_parameters_from_iters(
      std::vector<einsum_ir::basic::iter_property> const & iters,
      std::vector<dim_t>                                  & dim_types,
      std::vector<exec_t>                                 & exec_types,
      std::vector<int64_t>                                & dim_sizes,
      std::vector<int64_t>                                & strides_in0,
      std::vector<int64_t>                                & strides_in1,
      std::vector<int64_t>                                & strides_out,
      std::vector<int64_t>                                & packing_in0,
      std::vector<int64_t>                                & packing_in1
    );

    /**
     * Converts TensorOperation dtype to number of bytes.
     *
     * @param dtype TensorOperation data type.
     * @return      Number of bytes for the given data type.
     **/
    static inline int64_t dtype_to_num_bytes( dtype_t dtype );

    /**
     * Determines operation type based on primitive type.
     *
     * @param prim_main Main primitive type.
     * @return          Operation type (binary, unary, or undefined).
     **/
    static inline op_type_t determine_op_type( prim_t prim_main );

    /**
     * Determines number of threads (handles OpenMP detection).
     *
     * @param num_threads Requested threads (<=0 means auto-detect).
     * @return            Actual number of threads to use.
     **/
    static inline int64_t get_num_threads( int64_t num_threads );

    /**
     * Calculate SFC dimension sizes from configuration.
     *
     * Iterates through dim_types and exec_types to find SFC dimensions
     * and compute their combined sizes for M and N dimensions.
     *
     * @param dim_types    Dimension types vector.
     * @param exec_types   Execution types vector.
     * @param dim_sizes    Dimension sizes vector.
     * @param o_size_sfc_m Combined size of all SFC M dimensions.
     * @param o_size_sfc_n Combined size of all SFC N dimensions.
     **/
    static void calculate_sfc_sizes(
      std::vector< dim_t >   const & dim_types,
      std::vector< exec_t >  const & exec_types,
      std::vector< int64_t > const & dim_sizes,
      int64_t                      & o_size_sfc_m,
      int64_t                      & o_size_sfc_n
    );

    /**
     * Setup for unary operations.
     *
     * @param dtype       Datatype of tensor elements.
     * @param prim_main   Type of the main primitive.
     * @param exec_types  Execution types.
     * @param dim_sizes   Sizes of dimensions.
     * @param strides_in0 Strides of input tensor.
     * @param strides_out Strides of output tensor.
     * @return            Appropriate error code.
     **/
    error_t setup_unary(
      dtype_t                        dtype,
      prim_t                         prim_main,
      std::vector< exec_t >  const & exec_types,
      std::vector< int64_t > const & dim_sizes,
      std::vector< int64_t > const & strides_in0,
      std::vector< int64_t > const & strides_out
    );

    /**
     * Setup for binary operations.
     *
     * @param dtype       Datatype of tensor elements.
     * @param prim_first  Type of the first touch primitive.
     * @param prim_main   Type of the main primitive.
     * @param prim_last   Type of the last touch primitive.
     * @param dim_types   Dimension types.
     * @param exec_types  Execution types.
     * @param dim_sizes   Sizes of dimensions.
     * @param strides     3D stride tensor [LEVEL][TENSOR][DIMENSION].
     * @return            Appropriate error code.
     **/
    error_t setup_binary(
      dtype_t                                                      dtype,
      prim_t                                                       prim_first,
      prim_t                                                       prim_main,
      prim_t                                                       prim_last,
      std::vector< dim_t >                                 const & dim_types,
      std::vector< exec_t >                                const & exec_types,
      std::vector< int64_t >                               const & dim_sizes,
      std::vector< std::vector< std::vector< int64_t > > > const & strides
    );

    /**
     * Optimize unary operation configuration.
     *
     * @param dtype         Datatype of tensor elements.
     * @param prim_main     Type of the main primitive.
     * @param dim_types     Dimension types.
     * @param exec_types    Execution types.
     * @param dim_sizes     Sizes of dimensions.
     * @param strides_in0   Strides of input tensor.
     * @param strides_out   Strides of output tensor.
     * @param num_threads   Number of threads.
     * @return              Appropriate error code.
     **/
    static error_t optimize_unary(
      dtype_t                  dtype,
      prim_t                 & prim_main,
      std::vector< dim_t >   & dim_types,
      std::vector< exec_t >  & exec_types,
      std::vector< int64_t > & dim_sizes,
      std::vector< int64_t > & strides_in0,
      std::vector< int64_t > & strides_out,
      int64_t                  num_threads
    );

    /**
     * Optimize binary operation configuration.
     *
     * @param dtype               Datatype of tensor elements.
     * @param prim_first          Type of the first touch primitive.
     * @param prim_main           Type of the main primitive.
     * @param prim_last           Type of the last touch primitive.
     * @param dim_types           Dimension types.
     * @param exec_types          Execution types.
     * @param dim_sizes           Sizes of dimensions.
     * @param strides_in0         Strides of first input tensor.
     * @param strides_in1         Strides of second input tensor.
     * @param strides_out         Strides of output tensor.
     * @param packing_in0         Packing strides for first input tensor (output).
     * @param packing_in1         Packing strides for second input tensor (output).
     * @param target_m            Target M block size.
     * @param target_n            Target N block size.
     * @param target_k            Target K block size.
     * @param num_threads         Number of threads: [0]: shared, [1]: SFC M, [2]: SFC N.
     * @param packed_gemm_support Whether to enable packed GEMM support.
     * @param br_gemm_support     Whether to enable batch-reduce GEMM support.
     * @param packing_support     Whether to enable packing support.
     * @param sfc_support         Whether to enable SFC support.
     * @param l2_cache_size       Size of L2 cache in bytes.
     * @return                    Appropriate error code.
     **/
    static error_t optimize_binary(
      dtype_t                  dtype,
      prim_t                 & prim_main,
      std::vector< dim_t >   & dim_types,
      std::vector< exec_t >  & exec_types,
      std::vector< int64_t > & dim_sizes,
      std::vector< int64_t > & strides_in0,
      std::vector< int64_t > & strides_in1,
      std::vector< int64_t > & strides_out,
      std::vector< int64_t > & packing_in0,
      std::vector< int64_t > & packing_in1,
      int64_t                  target_m,
      int64_t                  target_n,
      int64_t                  target_k,
      int64_t                  num_threads[3],
      bool                     packed_gemm_support,
      bool                     br_gemm_support,
      bool                     packing_support,
      bool                     sfc_support,
      int64_t                  l2_cache_size
    );
};

#endif
