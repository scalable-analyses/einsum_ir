#ifndef EINSUM_IR_BACKEND_UNARY_LOOPS
#define EINSUM_IR_BACKEND_UNARY_LOOPS

#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class UnaryLoops;
  }
}

class einsum_ir::backend::UnaryLoops {

  public:
    //! number of dimensions
    int64_t m_num_dims = 0;

    //! sizes of the dimensions
    int64_t const * m_sizes = nullptr;

    //! strides of the input tensor
    int64_t const * m_strides_in = nullptr;
    //! strides of the output tensor
    int64_t const * m_strides_out = nullptr;

    //! number of bytes for a scalar of the input tensor
    int64_t m_num_bytes_scalar_in = 0;
    //! number of bytes for a scalar of the output tensor
    int64_t m_num_bytes_scalar_out = 0;

    //! number of loops used for threading
    int64_t m_threading_num_loops = -1;

    //! true if the unary loop interface was compiled
    bool m_compiled = false;

    /**
     * Kernel called in the innermost loop.
     *
     * @param i_ptr_in pointer to a data section of the input tensor.
     * @param io_ptr_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_main( void const * i_ptr_in,
                              void       * io_ptr_out ) = 0;

    /**
     * Initializes the loop-based implementation of a unary operator.
     *
     * @param i_num_dims number of dimensions.
     * @param i_sizes sizes of the dimensions.
     * @param i_strides_in strides of the input tensor.
     * @param i_strides_out strides of the output tensor.
     * @param i_num_bytes_in number of bytes per scalar in the input tensor.
     * @param i_num_bytes_out number of bytes per scalar in the output tensor.
     */
    void init( int64_t         i_num_dims,
               int64_t const * i_sizes,
               int64_t const * i_strides_in,
               int64_t const * i_strides_out,
               int64_t         i_num_bytes_in,
               int64_t         i_num_bytes_out );

    /**
     * Compiles the unary loop interface.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile();

    /**
     * Derives the threading data for the unary loops.
     * Parallelizes all loops such that the targeted number of tasks is reached or
     * all paralellizable loop dimensions have been exhausted.
     *
     * @param i_num_tasks_target number of targeted tasks. 
     **/
    void threading( int64_t i_num_tasks_target );

    /**
     * General purpose implementation which evaluates the unary operator.
     * No threading is applied.
     *
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_in pointer to the input tensor's data.
     * @param io_ptr_out pointer to the output tensor's data.
     **/
    void eval_iter( int64_t      i_id_loop,
                    void const * i_ptr_in,
                    void       * io_ptr_out );

    /**
     * Threaded implementation which evaluates the unary operator.
     * The outermost loop is parallelized, then eval_iter is called.
     *
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_in pointer to the input tensor's data.
     * @param io_ptr_out pointer to the output tensor's data.
     **/
    void eval_iter_parallel_1( int64_t      i_id_loop,
                               void const * i_ptr_in,
                               void       * io_ptr_out );

    /**
     * Threaded implementation which evaluates the unary operator.
     * The 2x outermost loop are parallelized (collapsed), then eval_iter is called.
     *
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_in pointer to the input tensor's data.
     * @param io_ptr_out pointer to the output tensor's data.
     **/
    void eval_iter_parallel_2( int64_t      i_id_loop,
                               void const * i_ptr_in,
                               void       * io_ptr_out );

    /**
     * Threaded implementation which evaluates the unary operator.
     * The 3x outermost loop are parallelized (collapsed), then eval_iter is called.
     *
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_in pointer to the input tensor's data.
     * @param io_ptr_out pointer to the output tensor's data.
     **/
    void eval_iter_parallel_3( int64_t      i_id_loop,
                               void const * i_ptr_in,
                               void       * io_ptr_out );

    /**
     * Threaded implementation which evaluates the unary operator.
     * The 4x outermost loop are parallelized (collapsed), then eval_iter is called.
     *
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_in pointer to the input tensor's data.
     * @param io_ptr_out pointer to the output tensor's data.
     **/
    void eval_iter_parallel_4( int64_t      i_id_loop,
                               void const * i_ptr_in,
                               void       * io_ptr_out );

    /**
     * Evaluates the unary operator.
     *
     * @param i_tensor_in input tensor.
     * @param io_tensor_out output tensor.
     **/
    void eval( void const * i_tensor_in,
               void       * io_tensor_out );
};

#endif