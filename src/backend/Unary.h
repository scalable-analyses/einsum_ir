#ifndef EINSUM_IR_BACKEND_UNARY
#define EINSUM_IR_BACKEND_UNARY

#include <cstdint>
#include <vector>
#include <map>
#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class Unary;
  }
}

class einsum_ir::backend::Unary {

  public:
    //! number of dimensions
    int64_t m_num_dims = 0;

    //! mapping from the dimension ids to the dimension sizes
    std::map< int64_t, int64_t > const * m_dim_sizes = nullptr;

    //! dimension ids of the input tensor
    int64_t const * m_dim_ids_in = nullptr;
    //! dimension ids of the ouput tensor
    int64_t const * m_dim_ids_out= nullptr;

    //! sizes of the output tensor
    std::vector< int64_t > m_sizes_out;

    //! strides of the input tensor w.r.t. to dimension ordering of the output tensor
    std::vector< int64_t > m_strides_in;
    //! strides of the output tensor
    std::vector< int64_t > m_strides_out;

    //! datatype of the input
    data_t m_dtype_in = UNDEFINED_DTYPE;

    //! datatype used during the computations
    data_t m_dtype_comp = UNDEFINED_DTYPE;

    //! datatype of the output
    data_t m_dtype_out = UNDEFINED_DTYPE;

    //! type of the main kernel
    kernel_t m_ktype_main = UNDEFINED_KTYPE;

    //! true if the unary operation was compiled
    bool m_compiled = false;

    /**
     * Derives the strides based on the sizes of the dimensions in the respective tensors.
     *
     * @param i_num_dims number of dimensions.
     * @param i_dim_sizes dimension sizes.
     * @param i_dim_ids dimension ids of the tensor.
     * @param o_strides will be set to the strides of the tensor.
     **/
    static void strides( int64_t                              i_num_dims,
                         std::map< int64_t, int64_t > const * i_dim_sizes,
                         int64_t                      const * i_dim_ids,
                         int64_t                            * o_strides);

    /**
     * Reorders the strides of the input tensor based on the dimensions of the output tensor.
     *
     * @param i_num_dims number of dimensions.
     * @param i_dim_ids_in dimension ids of the input tensor.
     * @param i_dim_ids_out dimension ids of the output tensor.
     * @param io_strides will be set to the reordered strides of the tensor.
     **/
    static void order_strides_output_based( int64_t         i_num_dims,
                                            int64_t const * i_dim_ids_in,
                                            int64_t const * i_dim_ids_out,
                                            int64_t       * io_strides);

    /**
     * Virtual destructor.
     **/
    virtual ~Unary(){};

    /**
     * Initializes the unary operation.
     *
     * @param i_num_dims number of dimensions.
     * @param i_dim_sizes dimension sizes.
     * @param i_dim_ids_in dimension ids of the input tensor.
     * @param i_dim_ids_out dimension ids of the output tensor.
     * @param o_strides_in will be set to the strides of the input tensor.
     * @param o_strides_out will be set to the strides of the output tensor.
     * @param i_dtype_in datatype of the input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_main type of the main kernel.
     **/
    void init( int64_t                              i_num_dims,
               std::map< int64_t, int64_t > const * i_dim_sizes,
               int64_t                      const * i_dim_ids_in,
               int64_t                      const * i_dim_ids_out,
               data_t                               i_dtype_in,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_main );


    /**
     * Initializes the unary operation with predefined strides
     *
     * @param i_num_dims number of dimensions.
     * @param i_dim_sizes dimension sizes.
     * @param i_dim_ids_in dimension ids of the input tensor.
     * @param i_dim_ids_out dimension ids of the output tensor.
     * @param i_strides_in will be set to the strides of the input tensor.
     * @param i_strides_out will be set to the strides of the output tensor.
     * @param i_dtype_in datatype of the input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_main type of the main kernel.
     **/
    void init( int64_t                              i_num_dims,
               std::map< int64_t, int64_t > const * i_dim_sizes,
               int64_t                      const * i_dim_ids_in,
               int64_t                      const * i_dim_ids_out,
               int64_t                            * i_strides_in,
               int64_t                            * i_strides_out,
               data_t                               i_dtype_in,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_main );

    /**
     * Compiles the base data of the unary operation.
     *
     * @return SUCCESS if successful, error code otherwise.
     **/
    err_t compile_base();

    /**
     * Compiles the unary operation. 
     *
     * @return SUCCESS if successful, error code otherwise.
     **/
    virtual err_t compile() = 0;

    /**
     * Evaluates the unary operation on the given input data.
     *
     * @param i_tensor_in input tensor.
     * @param io_tensor_out output tensor. 
     **/
    virtual void eval( void const * i_tensor_in,
                       void       * io_tensor_out ) = 0;

    /**
     * Initializes the threading configuration of the unary operation.
     *
     * @param i_num_tasks_target number of targeted tasks.
     **/
    virtual void threading( int64_t i_num_tasks_target  ) = 0;
};

#endif