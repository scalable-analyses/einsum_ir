#ifndef EINSUM_IR_BASIC_BINARY_CONTRACTION_BACKEND
#define EINSUM_IR_BASIC_BINARY_CONTRACTION_BACKEND

#include <vector>

#include "../constants.h"
#include "IterationSpace.h"
#include "ContractionMemoryManager.h"
#include "../unary/UnaryBackendTpp.h"


namespace einsum_ir {
  namespace basic {
    class ContractionBackend;
  }
}

class einsum_ir::basic::ContractionBackend {
  private:
    //! Iteration Space for parallel execution
    IterationSpace m_iter;

    //! id of the first parallelized loop
    int64_t m_id_first_parallel_loop = 0;

    //! id of the first primitive loop
    int64_t m_id_first_primitive_dim = 0;

    //! number of sfc loops
    int64_t m_num_sfc_loops = 0;

    //! number of omp loops
    int64_t m_num_omp_loops = 0;

    //! number of threads used for execution
    int64_t m_num_threads = 0;

    //! number of threads used for sfc m dimension
    int64_t m_num_threads_m = 0;
    //! number of threads used for sfc n dimension
    int64_t m_num_threads_n = 0;
    //! number of threads used for omp dimension
    int64_t m_num_threads_omp = 0;

    //! indicates existance of first touch kernel
    bool m_has_first_touch = false;

    //! indicates existance of last touch kernel
    bool m_has_last_touch = false;

    //! vector with thread personal information
    std::vector<thread_info> m_thread_infos;

    //! indicates if the backend is compiled
    bool m_is_compiled = false;

    //! pointer to active memory manager for contraction
    ContractionMemoryManager * m_memory = nullptr;

    //! personal memory manager for contraction, used if no external memory manager is given
    ContractionMemoryManager m_personal_memory;

    //! size of packed left input tensor
    int64_t m_size_packing_left  = 0;

    //! size of packed right input tensor
    int64_t m_size_packing_right = 0;

    //! unary packing backend for left input tensor
    UnaryBackendTpp m_unary_left;
    //! unary packing backend for right input tensor
    UnaryBackendTpp m_unary_right;

    //! id of the left packing loop
    int64_t m_packing_left_id  = -1;
    //! id of the right packing loop;
    int64_t m_packing_right_id = -1;

    //! number of cached pointers for left input tensor
    int64_t m_num_cached_ptrs_left  = 1;
    //! number of cached pointers for right input tensor
    int64_t m_num_cached_ptrs_right = 1;  

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

    //! vector with the packing strides of left tensor
    std::vector< int64_t > m_packing_strides_left;

    //! vector with the packing strides of right tensor
    std::vector< int64_t > m_packing_strides_right;

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

    //! kernel packed stride a
    uint64_t m_packed_stride_a = 0;
    //! kernel packed stride b
    uint64_t m_packed_stride_b = 0;

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

    //! vector of function pointers to the loop implementations
    std::vector<void (ContractionBackend::*)( thread_info *,
                                              int64_t,
                                              char const  *,
                                              char const  *,
                                              char const  *,
                                              char        *,
                                              bool,
                                              bool) > m_loop_functs;
    
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
     * @param i_packing_strides_left strides for packing of left input tensor.
     * @param i_packing_strides_right strides for packing of right input tensor.
     * @param i_dtype_left datatype of left input tensor.
     * @param i_dtype_right datatype of right input tensor.
     * @param i_dtype_comp datatype of computation.
     * @param i_dtype_out datatype of output tensor.
     * @param i_ktype_first_touch type of the first touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     * @param i_num_threads_omp number of threads used for omp parallelization.
     * @param i_num_threads_m number of threads used for sfc m parallelization.
     * @param i_num_threads_n number of threads used for sfc n parallelization.
     * @param i_contraction_mem pointer to the contraction memory manager.
     **/
    void init( std::vector< dim_t >   const & i_dim_type,
               std::vector< exec_t >  const & i_exec_type,
               std::vector< int64_t > const & i_dim_sizes,
               std::vector< int64_t > const & i_strides_left,
               std::vector< int64_t > const & i_strides_right,
               std::vector< int64_t > const & i_strides_out_aux,
               std::vector< int64_t > const & i_strides_out,
               std::vector< int64_t > const & i_packing_strides_left,
               std::vector< int64_t > const & i_packing_strides_right,
               data_t                         i_dtype_left,
               data_t                         i_dtype_right,
               data_t                         i_dtype_comp,
               data_t                         i_dtype_out,
               kernel_t                       i_ktype_first_touch,
               kernel_t                       i_ktype_main,
               kernel_t                       i_ktype_last_touch,
               int64_t                        i_num_threads_omp,
               int64_t                        i_num_threads_m,
               int64_t                        i_num_threads_n,
               ContractionMemoryManager     * i_contraction_mem );


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
     * @param i_num_threads_omp number of threads used for omp parallelization.
     * @param i_num_threads_m number of threads used for sfc m parallelization.
     * @param i_num_threads_n number of threads used for sfc n parallelization.
     * @param i_contraction_mem pointer to the contraction memory manager.
     **/
    void init( std::vector< iter_property > const & i_iterations,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch,
               int64_t                              i_num_threads_omp,
               int64_t                              i_num_threads_m,
               int64_t                              i_num_threads_n,
               ContractionMemoryManager           * i_contraction_mem );

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
     * @param i_thread_inf information for the executing thread.
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_left pointer to the left tensor's data.
     * @param i_ptr_right pointer to the right tensor's data.
     * @param i_ptr_out_aux pointer to the auxiliary output tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     * @param i_first_access true if first time accessing this data
     * @param i_last_access true if last time accessing this data
     **/
    void contract_iter( thread_info   * i_thread_inf,
                        int64_t         i_id_loop,
                        char    const * i_ptr_left,
                        char    const * i_ptr_right,
                        char    const * i_ptr_out_aux,
                        char          * i_ptr_out,
                        bool            i_first_access,
                        bool            i_last_access );

    /**
     * General purpose loop implementation featuring first and last touch operations.
     * Threading is applied.
     *
     * @param i_thread_inf information for the executing thread.
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_left pointer to the left tensor's data.
     * @param i_ptr_right pointer to the right tensor's data.
     * @param i_ptr_out_aux pointer to the auxiliary output tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     * @param i_first_access true if first time accessing this data
     * @param i_last_access true if last time accessing this data
     **/
    void contract_iter_parallel( thread_info   * i_thread_inf,
                                 int64_t         i_id_loop,
                                 char    const * i_ptr_left,
                                 char    const * i_ptr_right,
                                 char    const * i_ptr_out_aux,
                                 char          * i_ptr_out,
                                 bool            i_first_access,
                                 bool            i_last_access );
 
    /**
     * SFC based loop implementation featuring an featuring first and last touch operations.
     *
     * @param i_thread_inf information for the executing thread.
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_left pointer to the left tensor's data.
     * @param i_ptr_right pointer to the right tensor's data.
     * @param i_ptr_out_aux pointer to the auxiliary output tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     * @param i_first_access true if first time accessing this data
     * @param i_last_access true if last time accessing this data
     **/
    void contract_iter_sfc( thread_info   * i_thread_inf,
                            int64_t         i_id_loop,
                            char    const * i_ptr_left,
                            char    const * i_ptr_right,
                            char    const * i_ptr_out_aux,
                            char          * i_ptr_out,
                            bool            i_first_access,
                            bool            i_last_access );

    /**
     * Inner most loop implementation based on kernel call featuring first and last touch operations.
     *
     * @param i_thread_inf information for the executing thread.
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_left pointer to the left tensor's data.
     * @param i_ptr_right pointer to the right tensor's data.
     * @param i_ptr_out_aux pointer to the auxiliary output tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     * @param i_first_access true if first time accessing this data
     * @param i_last_access true if last time accessing this data
     **/
    void contract_iter_kernel( thread_info   * i_thread_inf,
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
     * Creates a packing operation for the left or right tensor.
     *
     * @param o_packing_id id of the packing loop.
     * @param o_size_packing memory size requried for packing.
     * @param o_unary compiled unary backend used for packing.
     * @param i_strides strides of the input tensor.
     * @param i_packing_strides strides of the packing tensor.
     *
     * @return SUCCESS if packing was created successfully, otherwise an appropiate error code.
     **/
    err_t create_packing( int64_t              & o_packing_id,
                          int64_t              & o_size_packing,
                          UnaryBackendTpp      & o_unary,
                          std::vector<int64_t> & i_strides,
                          std::vector<int64_t> & i_packing_strides );

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