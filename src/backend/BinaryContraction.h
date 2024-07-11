#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION

#include <cstdint>
#include <vector>
#include <map>
#include "../constants.h"
#include "MemoryManager.h"

namespace einsum_ir {
  namespace backend {
    class BinaryContraction;
  }
}

class einsum_ir::backend::BinaryContraction {
  public:
    //! number of dimensions of the left tensor
    int64_t m_num_dims_left = 0;
    //! number of dimensions of the right tensor
    int64_t m_num_dims_right = 0;
    //! number of dimensions of the output tensor
    int64_t m_num_dims_out = 0;

    //! mapping from the dimension ids to the inner dimension sizes
    std::map< int64_t, int64_t > const * m_dim_sizes_inner = nullptr;
    //! mapping from the dimension ids to the outer dimension sizes of the left tensor
    std::map< int64_t, int64_t > const * m_dim_sizes_outer_left = nullptr;
    //! mapping from the dimension ids to the outer dimension sizes of the right tensor
    std::map< int64_t, int64_t > const * m_dim_sizes_outer_right = nullptr;
    //! mapping from the dimension ids to the outer dimension sizes of the auxiliary output tensor
    std::map< int64_t, int64_t > const * m_dim_sizes_outer_out_aux = nullptr;
    //! mapping from the dimension ids to the outer dimension sizes of the output tensor
    std::map< int64_t, int64_t > const * m_dim_sizes_outer_out = nullptr;
    
    //! mapping from the dimension ids to dimension types
    std::map< int64_t, dim_t > m_dim_types;

    //! left tensor's dimension ids
    int64_t const * m_dim_ids_left = nullptr;
    //! right tensor's dimension ids
    int64_t const * m_dim_ids_right = nullptr;
    //! output tensor's dimension ids
    int64_t const * m_dim_ids_out = nullptr;

    //! permutation of dimension for the left tensor
    int64_t const * m_dim_ids_permute_left = nullptr;
    //! permutation of dimension for the right tensor
    int64_t const * m_dim_ids_permute_right = nullptr;

    //! dimension types of the output tensor
    std::vector< dim_t > m_dim_types_out;

    //! number of C dimensions
    int64_t m_num_dims_c = 0;
    //! number of M dimensions
    int64_t m_num_dims_m = 0;
    //! number of N dimensions
    int64_t m_num_dims_n = 0;
    //! number of K dimensions
    int64_t m_num_dims_k = 0;
    //! number of I dimensions
    int64_t m_num_dims_i = 0;
    //! number of J dimensions
    int64_t m_num_dims_j = 0;

    //! C dimension ids
    std::vector< int64_t > m_dim_ids_c;
    //! M dimension ids
    std::vector< int64_t > m_dim_ids_m;
    //! N dimension ids
    std::vector< int64_t > m_dim_ids_n;
    //! K dimension ids
    std::vector< int64_t > m_dim_ids_k;
    //! I dimension ids
    std::vector< int64_t > m_dim_ids_i;
    //! J dimension ids
    std::vector< int64_t > m_dim_ids_j;

    //! sizes of the C dimensions
    std::vector< int64_t > m_sizes_c;
    //! sizes of the M dimensions
    std::vector< int64_t > m_sizes_m;
    //! sizes of the N dimensions
    std::vector< int64_t > m_sizes_n;
    //! sizes of the K dimensions
    std::vector< int64_t > m_sizes_k;
    //! sizes of the I dimensions
    std::vector< int64_t > m_sizes_i;
    //! sizes of the J dimensions
    std::vector< int64_t > m_sizes_j;

    //! datatype of the left input
    data_t m_dtype_left = UNDEFINED_DTYPE;

    //! datatype of the right input
    data_t m_dtype_right = UNDEFINED_DTYPE;

    //! datatype used during the computations
    data_t m_dtype_comp = UNDEFINED_DTYPE;

    //! datatype of the output
    data_t m_dtype_out = UNDEFINED_DTYPE;

    //! type of the first touch kernel
    kernel_t m_ktype_first_touch = UNDEFINED_KTYPE;

    //! type of the main kernel
    kernel_t m_ktype_main = UNDEFINED_KTYPE;

    //! type of the last touch kernel
    kernel_t m_ktype_last_touch = UNDEFINED_KTYPE;

    //! true if the binary contraction was compiled
    bool m_compiled = false;

    //! Memory manager for intermendiate results
    MemoryManager * m_memory = nullptr;

    /**
     * Derives the dimension types of tensor t2 w.r.t. tensors t0 and t1.
     *
     * @param i_num_dims_t0 number of t0 dimensions.
     * @param i_num_dims_t1 number of t1 dimensions.
     * @param i_num_dims_t2 number of t2 dimensions.
     * @param i_dim_ids_t0 dimension identifiers of t0's dimensions.
     * @param i_dim_ids_t1 dimension identifiers of t1's dimensions.
     * @param i_dim_ids_t2 dimension identifiers of t2's dimensions.
     * @param i_dim_type_t2 dimension type which will be set if dimension is exclusive to t2.
     * @param i_dim_type_t2_t0 dimension type which will be set if dimension is part of t2 and t0.
     * @param i_dim_type_t2_t1 dimension type which will be set if dimension is part of t2 and t1.
     * @param i_dim_type_t2_t0_t1 dimension type which will be set if dimension is part of t2, t0 and t1.
     * @param o_dim_types_t2 will be set to dimension types of t2.
     **/
    static void dim_types( int64_t         i_num_dims_t0,
                           int64_t         i_num_dims_t1,
                           int64_t         i_num_dims_t2,
                           int64_t const * i_dim_ids_t0,
                           int64_t const * i_dim_ids_t1,
                           int64_t const * i_dim_ids_t2,
                           dim_t           i_dim_type_t2,
                           dim_t           i_dim_type_t2_t0,
                           dim_t           i_dim_type_t2_t1,
                           dim_t           i_dim_type_t2_t0_t1,
                           dim_t         * o_dim_types_t2 );

    /**
     * Derives the dimension types of the left, right and output tensor w.r.t. the binary contraction.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param o_dim_types_left will be set to dimension types of the left tensor.
     * @param o_dim_types_right will be set to dimension types of the right tensor.
     * @param o_dim_types_out will be set to dimension types of the output tensor.
     **/
    static void dim_types( int64_t                                 i_num_dims_left,
                           int64_t                                 i_num_dims_right,
                           int64_t                                 i_num_dims_out,
                           int64_t                         const * i_dim_ids_left,
                           int64_t                         const * i_dim_ids_right,
                           int64_t                         const * i_dim_ids_out,
                           std::vector< einsum_ir::dim_t >       * o_dim_types_left,
                           std::vector< einsum_ir::dim_t >       * o_dim_types_right,
                           std::vector< einsum_ir::dim_t >       * o_dim_types_out );

    /**
     * Filters the dimension ids based on the given dimension type.
     *
     * @param i_num_dims_tensor number of tensor dimensions.
     * @param i_dim_type_filter dimension type which is used as filter.
     * @param i_dim_types_tensor types of the tensor's dimensions.
     * @param i_dim_ids_tensor ids of the tensor's dimensions.
     * @param o_dim_ids_filtered will be set to filtered dimension ids.
     * @return number of found dimension matching the given type. 
     **/
    static int64_t filter_dim_ids( int64_t         i_num_dims_tensor,
                                   dim_t           i_dim_type_filter,
                                   dim_t   const * i_dim_types_tensor,
                                   int64_t const * i_dim_ids_tensor,
                                   int64_t       * o_dim_ids_filtered );

    /**
     * Derives the dimension types and ids.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param o_dim_types_out will be set to dimension types of the output tensor.
     * @param o_dim_ids_c will be set to ids of the C dimensions.
     * @param o_dim_ids_m will be set to ids of the M dimensions.
     * @param o_dim_ids_n will be set to ids of the N dimensions.
     * @param o_dim_ids_k will be set to ids of the K dimensions.
     * @param o_dim_ids_i will be set to ids of the I dimensions.
     * @param o_dim_ids_j will be set to ids of the J dimensions.
     **/
    static void dim_types_ids( int64_t                                 i_num_dims_left,
                               int64_t                                 i_num_dims_right,
                               int64_t                                 i_num_dims_out,
                               int64_t                         const * i_dim_ids_left,
                               int64_t                         const * i_dim_ids_right,
                               int64_t                         const * i_dim_ids_out,
                               std::vector< einsum_ir::dim_t >       * o_dim_types_out,
                               std::vector<          int64_t >       * o_dim_ids_c,
                               std::vector<          int64_t >       * o_dim_ids_m,
                               std::vector<          int64_t >       * o_dim_ids_n,
                               std::vector<          int64_t >       * o_dim_ids_k,
                               std::vector<          int64_t >       * o_dim_ids_i,
                               std::vector<          int64_t >       * o_dim_ids_j );

    /**
     * Derives the strides for the dimensions in a tensor.
     *
     * @param i_num_dims number of tensor dimensions.
     * @param i_dim_ids dimension ids of the tensor.
     * @param i_dim_sizes key-value (dim_id-size) sizes of the dimensions.
     * @param o_strides will be set set to key-value (dim_id-stride) strides of the dimensions.
     **/
    static void strides( int64_t                              i_num_dims,
                         int64_t                      const * i_dim_ids,
                         std::map< int64_t, int64_t > const * i_dim_sizes,
                         std::map< int64_t, int64_t >       * o_strides );

    /**
     * Virtual destructor.
     **/
    virtual ~BinaryContraction(){};

    /**
     * Initializes the binary contraction.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_sizes_inner mapping from the dimension ids to the inner sizes.
     * @param i_dim_sizes_outer_left mapping from the dimension ids to the left tensor's outer sizes.
     * @param i_dim_sizes_outer_right mapping from the dimension ids to the right tensor's outer sizes.
     * @param i_dim_sizes_outer_out_aux mapping from the dimension ids to the auxiliary out tensor's outer sizes.
     * @param i_dim_sizes_outer_out mapping from the dimension ids to the out tensor's outer sizes.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param i_dtype_left datatype of the left input.
     * @param i_dtype_right datatype of the right input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     **/
    void init( int64_t                              i_num_dims_left,
               int64_t                              i_num_dims_right,
               int64_t                              i_num_dims_out,
               std::map< int64_t, int64_t > const * i_dim_sizes_inner,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer_left,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer_right,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer_out_aux,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer_out,
               int64_t                      const * i_dim_ids_left,
               int64_t                      const * i_dim_ids_right,
               int64_t                      const * i_dim_ids_out,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch );

    /**
     * Initializes the binary contraction with fused permutations of the input tensors
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_sizes_inner mapping from the dimension ids to the inner sizes.
     * @param i_dim_sizes_outer_left mapping from the dimension ids to the left tensor's outer sizes.
     * @param i_dim_sizes_outer_right mapping from the dimension ids to the right tensor's outer sizes.
     * @param i_dim_sizes_outer_out_aux mapping from the dimension ids to the auxiliary out tensor's outer sizes.
     * @param i_dim_sizes_outer_out mapping from the dimension ids to the out tensor's outer sizes.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param i_dim_ids_permute_left permutation of dimension for the left tensor
     * @param i_dim_ids_permute_right permutation of dimension for the right tensor
     * @param i_memory memory manager for efficient memory usage.
     * @param i_dtype_left datatype of the left input.
     * @param i_dtype_right datatype of the right input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     **/
    void init( int64_t                              i_num_dims_left,
               int64_t                              i_num_dims_right,
               int64_t                              i_num_dims_out,
               std::map< int64_t, int64_t > const * i_dim_sizes_inner,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer_left,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer_right,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer_out_aux,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer_out,
               int64_t                      const * i_dim_ids_left,
               int64_t                      const * i_dim_ids_right,
               int64_t                      const * i_dim_ids_out,
               int64_t                      const * i_dim_ids_permute_left,
               int64_t                      const * i_dim_ids_permute_right,
               MemoryManager                      * i_memory,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch );

    /**
     * Compiles the base data.
     *
     * @return SUCCESS if successful, error code otherwise.
     **/
    err_t compile_base();

    /**
     * Compiles the binary contraction. 
     *
     * @return SUCCESS if successful, error code otherwise.
     **/
    virtual err_t compile() = 0;

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_left left input tensor.
     * @param i_tensor_right right input tensor.
     * @param io_tensor_out output tensor. 
     **/
    virtual void contract( void const * i_tensor_left,
                           void const * i_tensor_right,
                           void       * io_tensor_out ) = 0;

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_left left input tensor.
     * @param i_tensor_right right input tensor.
     * @param i_tensor_out_aux auxiliary data w.r.t. output tensor.
     * @param io_tensor_out output tensor.
     **/
    virtual void contract( void const * i_tensor_left,
                           void const * i_tensor_right,
                           void const * i_tensor_out_aux,
                           void       * io_tensor_out ) = 0;

    /**
     * Gets the number of operations for a single contraction.
     **/
    int64_t num_ops();

    /**
     * Initializes the threading configuration of the contraction.
     *
     * @param i_num_tasks_target number of targeted tasks.
     **/
    virtual void threading( int64_t i_num_tasks_target  ) = 0;
};

#endif