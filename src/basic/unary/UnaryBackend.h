#ifndef EINSUM_IR_BASIC_UNARY_BACKEND
#define EINSUM_IR_BASIC_UNARY_BACKEND

#include <vector>
#include "../constants.h"


namespace einsum_ir {
  namespace basic {
    class UnaryBackend;
  }
}

class einsum_ir::basic::UnaryBackend {
  private:

    //! id of the first primitive loop
    int64_t m_id_first_primitive_dim = 0;

    //! number of threads used for execution
    int64_t m_num_threads = 0;

    //! number of parallel loops
    int64_t m_num_parallel_loops = 0;

    //! id of the first parallel loop
    int64_t m_id_first_parallel_loop = 0;

  protected:
    //! datatype of the input
    data_t m_dtype_in = UNDEFINED_DTYPE;

    //! datatype of the output
    data_t m_dtype_out = UNDEFINED_DTYPE;

    //! datatype used during the computations
    data_t m_dtype_comp = UNDEFINED_DTYPE;


    //! vector with execution types of all loops
    std::vector< exec_t >   m_exec_type;

    //! vector with sizes of all loops
    std::vector< int64_t > m_dim_sizes;

    //! vector with the strides of the  input tensor
    std::vector< int64_t > m_strides_in;

    //! vector with the strides of the output tensor
    std::vector< int64_t > m_strides_out;

    //! type of the main kernel
    kernel_t  m_ktype = UNDEFINED_KTYPE;

    //! kernel m size
    uint64_t m_m = 0;
    //! kernel n size
    uint64_t m_n = 0;


    //! kernel leading dimension A
    uint64_t m_lda = 0;
    //! kernel leading dimension B
    uint64_t m_ldb = 0;

    //! indicates if kernel should transpose A
    bool m_trans_a = false;
    
  public:
    /**
     * Initializes the class.
     *
     * @param i_exec_type execution types.
     * @param i_dim_sizes sizes of the dimensions.
     * @param i_strides_in strides in the input tensor.
     * @param i_strides_out strides in the output tensor.
     * @param i_dtype_in datatype of left input tensor.
     * @param i_dtype_comp datatype of computation.
     * @param i_dtype_out datatype of output tensor.
     * @param i_ktype type of the kernel.
     * @param i_num_threads number of threads used for contraction.
     **/
    void init( std::vector< exec_t >  const & i_exec_type,
               std::vector< int64_t > const & i_dim_sizes,
               std::vector< int64_t > const & i_strides_in,
               std::vector< int64_t > const & i_strides_out,
               data_t                         i_dtype_in,
               data_t                         i_dtype_comp,
               data_t                         i_dtype_out,
               kernel_t                       i_ktype,
               int64_t                        i_num_threads );


    /**
     * Initializes the class with a vector of iter_properties.
     *
     * @param i_iterations vector of iter_properties.
     * @param i_dtype_in datatype of input tensor.
     * @param i_dtype_comp datatype of computation.
     * @param i_dtype_out datatype of output tensor.
     * @param i_ktype type of the kernel.
     * @param i_num_threads number of threads used for contraction.
     **/
    void init( std::vector< iter_property > const & i_iterations,
               data_t                               i_dtype_in,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype,
               int64_t                              i_num_threads );

    /**
     * Compiles the contraction loop interface.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile();


    /**
     * Contracts the two tensors.
     *
     * @param i_tensor_in input tensor.
     * @param io_tensor_out output tensor.
     **/
    void contract( void const * i_tensor_in,
                   void       * io_tensor_out );
    
    /**
     * General purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_in pointer to the input tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     **/
    void contract_iter( int64_t         i_id_loop,
                        char    const * i_ptr_in,
                        char          * i_ptr_out );

    /**
     * General purpose loop implementation featuring first and last touch operations.
     * Omp parallelization is applied by funsing all parallel loops.
     *
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_in pointer to the input tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     **/
    void contract_iter_parallel( int64_t         i_id_loop,
                                 char    const * i_ptr_in,
                                 char          * i_ptr_out );

    /**
     * calculates the shape of the kernel i.e. m, n, k, lda, ldb, ldc, ...
     *
     * @return SUCCESS if the shape matches with the main primitive, otherwise an appropiate error code.
     **/
    err_t set_kernel_shape( );

    /**
     * Kernel called in the innermost loop.
     *
     * @param i_in pointer to a data section of the tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_main( void const * i_in,
                              void       * io_out ) = 0;

    /**
     * Compiles all kernels
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    virtual err_t compile_kernels() = 0;
};

#endif