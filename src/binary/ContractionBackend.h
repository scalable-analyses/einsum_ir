#ifndef EINSUM_IR_BINARY_CONTRACTION_BACKEND
#define EINSUM_IR_BINARY_CONTRACTION_BACKEND

#include <vector>

#include "constants.h"
#include "IterationSpace.h"


namespace einsum_ir {
  namespace binary {
    class ContractionBackend;
  }
}

class einsum_ir::binary::ContractionBackend {
  private:
    //! Iteration Space for parallel execution
    einsum_ir::binary::IterationSpace m_iter;

    //! id of the first parallelized loop
    int64_t m_id_first_parallel_loop = 0;

    //! id of the first primitive loop
    int64_t m_id_first_primitive_dim = 0;

    //! number of parallel loops
    int64_t m_num_parallel_loops = 0;

    //! number of threads used for execution
    int64_t m_num_threads = 0;

    //! indicates existance of first touch kernel
    bool m_has_first_touch = false;

    //! indicates existance of last touch kernel
    bool m_has_last_touch = false;

  protected:
    //! datatype of the left input
    data_t m_dtype_left = UNDEFINED_DTYPE;

    //! datatype of the right input
    data_t m_dtype_right = UNDEFINED_DTYPE;

    //! datatype of the output
    data_t m_dtype_out = UNDEFINED_DTYPE;

    //! datatype used during the computations
    data_t m_dtype_comp = UNDEFINED_DTYPE;


    //! vector with dimension types of all loops
    std::vector< dim_t >   m_dim_type;

    //! vector with execution types of all loops
    std::vector< exec_t >  m_exec_type;

    //! vector with sizes of all loops
    std::vector< int64_t > m_dim_sizes;

    //! vector with the strides of the left input tensor
    std::vector< int64_t > m_strides_left;

    //! vector with the strides of the right input tensor
    std::vector< int64_t > m_strides_right;

    //! vector with the strides of the auxiliary tensor
    std::vector< int64_t > m_strides_out_aux;

    //! vector with the strides of the output tensor
    std::vector< int64_t > m_strides_out;

    //! type of the first touch kernel
    kernel_t m_ktype_first_touch = UNDEFINED_KTYPE;
    //! type of the main kernel
    kernel_t  m_ktype_main = UNDEFINED_KTYPE;
    //! type of the last touch kernel
    kernel_t m_ktype_last_touch = UNDEFINED_KTYPE;

    //! kernel br size
    uint64_t m_br = 0;
    //! kernel m size
    uint64_t m_m = 0;
    //! kernel n size
    uint64_t m_n = 0;
    //! kernel k size
    uint64_t m_k = 0;
    //! kernel r size
    uint64_t m_r = 0;


    //! kernel leading dimension A
    uint64_t m_lda = 0;
    //! kernel leading dimension B
    uint64_t m_ldb = 0;
    //! kernel leading dimension C
    uint64_t m_ldc = 0;

    //! kernel m stride of auxiliary tensor
    uint64_t m_stride_m_out_aux = 0;
    //! kernel n stride of auxiliary tensor
    uint64_t m_stride_n_out_aux = 0;

    //! kernel br stride a
    uint64_t m_br_stride_a = 0;
    //! kernel br stride b
    uint64_t m_br_stride_b = 0;

    //! complex stride of the left tensor
    int64_t m_cpx_stride_in_left_bytes = 0;
    //! complex stride of the right tensor
    int64_t m_cpx_stride_in_right_bytes = 0;
    //! complex stride of the auxiliary output tensor
    int64_t m_cpx_stride_out_aux_bytes = 0;
    //! complex stride of the output tensor
    int64_t m_cpx_stride_out_bytes = 0;

    //! indicates if kernel should transpose A
    bool m_trans_a = false;
    //! indicates if kernel should transpose B
    bool m_trans_b = false;
    
  public:
    /**
     * Initializes the class.
     *
     * @param i_dim_type dimension types.
     * @param i_exec_type execution types.
     * @param i_dim_sizes sizes of the dimensions.
     * @param i_strides_left strides in the left input tensor.
     * @param i_strides_right strides in the right input tensor.
     * @param i_strides_out_aux strides in the auxiliary output tensor.
     * @param i_strides_out strides in the output tensor.
     * @param i_dtype_left datatype of left input tensor.
     * @param i_dtype_right datatype of right input tensor.
     * @param i_dtype_comp datatype of computation.
     * @param i_dtype_out datatype of output tensor.
     * @param i_ktype_first_touch type of the first touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     * @param i_num_threads number of threads used for contraction.
     **/
    void init( std::vector< dim_t >   const & i_dim_type,
               std::vector< exec_t >  const & i_exec_type,
               std::vector< int64_t > const & i_dim_sizes,
               std::vector< int64_t > const & i_strides_left,
               std::vector< int64_t > const & i_strides_right,
               std::vector< int64_t > const & i_strides_out_aux,
               std::vector< int64_t > const & i_strides_out,
               data_t                         i_dtype_left,
               data_t                         i_dtype_right,
               data_t                         i_dtype_comp,
               data_t                         i_dtype_out,
               kernel_t                       i_ktype_first_touch,
               kernel_t                       i_ktype_main,
               kernel_t                       i_ktype_last_touch,
               int64_t                        i_num_threads );


    /**
     * Initializes the class with a vector of iter_properties.
     *
     * @param i_iterations vector of iter_properties.
     * @param i_dtype_left datatype of left input tensor.
     * @param i_dtype_right datatype of right input tensor.
     * @param i_dtype_comp datatype of computation.
     * @param i_dtype_out datatype of output tensor.
     * @param i_ktype_first_touch type of the first touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     * @param i_num_threads number of threads used for contraction.
     **/
    void init( std::vector< iter_property > const & i_iterations,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch,
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
     * @param i_tensor_left left tensor.
     * @param i_tensor_right right tensor.
     * @param i_tensor_out_aux auxiliary data w.r.t. output tensor.
     * @param io_tensor_out output tensor.
     **/
    void contract( void const * i_tensor_left,
                   void const * i_tensor_right,
                   void const * i_tensor_out_aux,
                   void       * io_tensor_out );
    
    /**
     * General purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param i_thread_id id of the executing thread
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_left pointer to the left tensor's data.
     * @param i_ptr_right pointer to the right tensor's data.
     * @param i_ptr_out_aux pointer to the auxiliary output tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     * @param i_first_access true if first time accessing this data
     * @param i_last_access true if last time accessing this data
     **/
    void contract_iter( int64_t         i_thread_id,
                        int64_t         i_id_loop,
                        char    const * i_ptr_left,
                        char    const * i_ptr_right,
                        char    const * i_ptr_out_aux,
                        char          * i_ptr_out,
                        bool            i_first_access,
                        bool            i_last_access );

    /**
     * calculates the shape of the kernel i.e. m, n, k, lda, ldb, ldc, ...
     *
     * @return SUCCESS if the shape matches with the main primitive, otherwise an appropiate error code.
     **/
    err_t set_kernel_shape( );

    /**
     * Kernel applied to the output tensor before the main primitive touches the memory.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_first_touch( void const * i_out_aux,
                                     void       * io_out ) = 0;

    /**
     * Kernel applied to the output tensor after the main primitve finished using the memory.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_last_touch( void const * i_out_aux,
                                    void       * io_out ) = 0;

    /**
     * Kernel called in the innermost loop.
     *
     * @param i_left pointer to a data section of the left tensor.
     * @param i_right pointer to a data section of the right tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_main( void const * i_left,
                              void const * i_right,
                              void       * io_out ) = 0;

    /**
     * Compiles all kernels
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    virtual err_t compile_kernels() = 0;
};

#endif