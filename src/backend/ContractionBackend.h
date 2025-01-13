#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS_SFC
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS_SFC

#include <vector>
#include <map>
#include "../constants.h"
#include "IterationSpacesSfc.h"


namespace einsum_ir {
  namespace backend {
    class ContractionBackend;
  }
}

class einsum_ir::backend::ContractionBackend {
  private:
    //! Iteration Space for parallel execution
    einsum_ir::backend::IterationSpacesSfc m_iter;

    //! id of the first parallelized loop
    int64_t m_id_first_parallel_loop = 0;

    //! id of the first primitive loop
    int64_t m_id_first_primitive_loop = 0;

    //! number of parallel loops
    int64_t m_num_parallel_loops = 0;

    //! indicates existance of first touch kernel
    bool m_has_first_touch = false;

    //! indicates existance of last touch kernel
    bool m_has_last_touch = false;

  protected:
    //! datatype of the left input
    data_t m_dtype_left = UNDEFINED_DTYPE;

    //! datatype of the right input
    data_t m_dtype_right = UNDEFINED_DTYPE;

    //! datatype used during the computations
    data_t m_dtype_comp = UNDEFINED_DTYPE;

    //! datatype of the output
    data_t m_dtype_out = UNDEFINED_DTYPE;

    //TODO 
    std::vector< dim_t >   m_loop_dim_type;
    std::vector< exec_t >  m_loop_exec_type;
    std::vector< int64_t > m_loop_sizes;
    std::vector< int64_t > m_loop_strides_left;
    std::vector< int64_t > m_loop_strides_right;
    std::vector< int64_t > m_loop_strides_out_aux;
    std::vector< int64_t > m_loop_strides_out;

    //! type of the first touch kernel
    kernel_t m_ktype_first_touch = UNDEFINED_KTYPE;
    //! type of the main kernel
    kernel_t m_ktype_main = UNDEFINED_KTYPE;
    //! type of the last touch kernel
    kernel_t m_ktype_last_touch = UNDEFINED_KTYPE;
    
  public:
    //TODO
    void init( std::vector< dim_t >   const & i_loop_dim_type,
               std::vector< exec_t >  const & i_loop_exec_type,
               std::vector< int64_t > const & i_loop_sizes,
               std::vector< int64_t > const & i_loop_strides_left,
               std::vector< int64_t > const & i_loop_strides_right,
               std::vector< int64_t > const & i_loop_strides_out_aux,
               std::vector< int64_t > const & i_loop_strides_out,
               data_t                         i_dtype_left,
               data_t                         i_dtype_right,
               data_t                         i_dtype_comp,
               data_t                         i_dtype_out,
               kernel_t                       i_ktype_first_touch,
               kernel_t                       i_ktype_main,
               kernel_t                       i_ktype_last_touch );


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
                        void    const * i_ptr_left,
                        void    const * i_ptr_right,
                        void    const * i_ptr_out_aux,
                        void          * i_ptr_out,
                        bool            i_first_access,
                        bool            i_last_access );


    /**
     * Kernel applied to the output tensor before the contraction.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_first_touch( void const * i_out_aux,
                                     void       * io_out ) = 0;

    /**
     * Kernel applied to the output tensor after the contraction.
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