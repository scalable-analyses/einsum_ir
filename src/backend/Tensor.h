#ifndef EINSUM_IR_BACKEND_TENSOR
#define EINSUM_IR_BACKEND_TENSOR

#include <cstdint>
#include <map>
#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class Tensor;
  }
}

class einsum_ir::backend::Tensor {
  private:
    /**
     * Transposes the given tensor.
     *
     * @paramt T_IN data type of the input data.
     * @paramt T_OUT data type of the output data.
     *
     * @param i_num_dims number of dimensions.
     * @param i_sizes sizes of the dimensions.
     * @param i_strides_in strides of the input tensor.
     * @param i_strides_out strides of the output tensor.
     * @param i_data_ptr_in data pointer of the input tensor.
     * @param o_data_ptr_out data pointer of the output tensor. 
     **/
    template< typename T_IN,
              typename T_OUT >
    static void transpose( int64_t         i_num_dims,
                           int64_t const * i_sizes,
                           int64_t const * i_strides_in,
                           int64_t const * i_strides_out,
                           T_IN    const * i_data_ptr_in,
                           T_OUT         * o_data_ptr_out );

  public:
    /**
     * Derives the size of the given tensor in bytes.
     *
     * @param i_bytes_per_entry number of bytes per entry in the tensor.
     * @param i_num_dims number of dimensions.
     * @param i_dim_ids dimension ids.
     * @param i_dim_sizes sizes of the dimension.
     **/
    static int64_t size( int64_t                              i_bytes_per_entry,
                         int64_t                              i_num_dims,
                         int64_t                      const * i_dim_ids,
                         std::map< int64_t, int64_t > const & i_dim_sizes );

    /**
     * Derives the permutation to transfer the input ids to the output ones.
     *
     * @param i_num_dims number of dimensions.
     * @param i_dim_ids_in dimension ids of the output tensor.
     * @param i_dim_ids_out dimension ids of the output tensor.
     * @param o_permutation will be set to derived permutation.
     **/
    static void permutation( int64_t         i_num_dims,
                             int64_t const * i_dim_ids_in,
                             int64_t const * i_dim_ids_out,
                             int64_t       * o_permutation );

    /**
     * Permutes the dimensions of the given tensor.
     *
     * @param i_num_dims number of dimensions.
     * @param i_dim_sizes sizes of the dimensions.
     * @param i_permutation permutation of the input tensor's dimensions w.r.t. the output tensor.
     * @param i_dtype_in data type of the input tensor.
     * @param i_dtype_out data type of the output tensor.
     * @param i_data_ptr_in data pointer of the input tensor.
     * @param i_data_ptr_out data pointer of the output tensor.  
     **/
    static void permute( int64_t         i_num_dims,
                         int64_t const * i_dim_sizes,
                         int64_t const * i_permutation,
                         data_t          i_dtype_in,
                         data_t          i_dtype_out,
                         void    const * i_data_ptr_in,
                         void          * o_tdata_ptr_out );

    /**
     * Permutes the dimensions of the given tensor.
     *
     * @param i_num_dims number of dimensions.
     * @param i_dim_sizes sizes of the dimensions.
     * @param i_dim_ids_in dimension ids of the input tensor.
     * @param i_dim_ids_out dimension ids of the output tensor.
     * @param i_dtype_in data type of the input tensor.
     * @param i_dtype_out data type of the output tensor.
     * @param i_data_ptr_in data pointer of the input tensor.
     * @param i_data_ptr_out data pointer of the output tensor.  
     **/
    static void permute( int64_t                              i_num_dims,
                         std::map< int64_t, int64_t > const & i_dim_sizes,
                         int64_t                      const * i_dim_ids_in,
                         int64_t                      const * i_dim_ids_out,
                         data_t                               i_dtype_in,
                         data_t                               i_dtype_out,
                         void                         const * i_data_ptr_in,
                         void                               * o_data_ptr_out );
};

#endif