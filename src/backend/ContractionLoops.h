#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS

#include <cstdint>
#include <vector>
#include "../constants.h"
#include "IterationSpaces.h"
#include "MemoryManager.h"

namespace einsum_ir {
  namespace backend {
    class ContractionLoops;
  }
}

class einsum_ir::backend::ContractionLoops {
  private:
    //! number of C dimensions
    int64_t m_num_dims_c = 0;
    //! number of M dimensions
    int64_t m_num_dims_m = 0;
    //! number of N dimensions
    int64_t m_num_dims_n = 0;
    //! number of K dimensions
    int64_t m_num_dims_k = 0;

    //! sizes of the C dimensions
    int64_t const * m_sizes_c = nullptr;
    //! sizes of the M dimensions
    int64_t const * m_sizes_m = nullptr;
    //! sizes of the N dimensions
    int64_t const * m_sizes_n = nullptr;
    //! sizes of the K dimensions
    int64_t const * m_sizes_k = nullptr;

    //! C strides of the left input tensor
    int64_t const * m_strides_in_left_c = nullptr;
    //! M strides of the left input tensor
    int64_t const * m_strides_in_left_m = nullptr;
    //! K strides of the left input tensor
    int64_t const * m_strides_in_left_k = nullptr;

    //! C strides of the right input tensor
    int64_t const * m_strides_in_right_c = nullptr;
    //! N strides of the right input tensor
    int64_t const * m_strides_in_right_n = nullptr;
    //! K strides of the right input tensor
    int64_t const * m_strides_in_right_k = nullptr;

    //! C strides of the auxiliary output tensor
    int64_t const * m_strides_out_aux_c = nullptr;
    //! M strides of the auxiliary output tensor
    int64_t const * m_strides_out_aux_m = nullptr;
    //! N strides of the auxiliary output tensor
    int64_t const * m_strides_out_aux_n = nullptr;

    //! C strides of the output tensor
    int64_t const * m_strides_out_c = nullptr;
    //! M strides of the output tensor
    int64_t const * m_strides_out_m = nullptr;
    //! N strides of the output tensor
    int64_t const * m_strides_out_n = nullptr;

    //! number of bytes for a scalar of the left input tensor
    int64_t m_num_bytes_scalar_left = 0;
    //! number of bytes for a scalar of the right input tensor
    int64_t m_num_bytes_scalar_right = 0;
    //! number of bytes for a scalar of the output tensor
    int64_t m_num_bytes_scalar_out = 0;

    //! number of targeted tasks
    int64_t m_num_tasks_targeted = 0;

    //! number of tasks
    int64_t m_num_tasks = 0;

    //! vector of memory pointers for thread specific packing memory
    MemoryManager * m_memory;

    //! amount of memory required for packing of the left tensor on one thread
    int64_t  m_size_packing_left;

    //! amount of memory required for packing of the right tensor on one thread
    int64_t  m_size_packing_right;

    //! vector with alllocated memory
    char ** m_memory_packing;

    //! iteration spaces
    IterationSpaces m_iter_spaces;

    //! true if the threading loops have to take care of fist/last touch ops
    bool m_threading_first_last_touch = false;

    //! first/last touch type of a loop
    typedef enum {
      // no first/last touch
      NONE = 0,
      // first touch before and last touch after the loop
      BEFORE_AFTER_ITER = 1,
      // first touch before the main kernel in every iteration
      // last touch after the main kernel in very iteration
      EVERY_ITER = 2
    } touch_t;

    //! number of loops
    int64_t m_num_loops = -1;
    //! first/last touch type of the loops
    std::vector< touch_t > m_loop_first_last_touch;
    //! dimension types of the loops (C, M, N or K)
    std::vector< dim_t >   m_loop_dim_type;
    //! sizes of the loops / number of iterations
    std::vector< int64_t > m_loop_sizes;
    //! per-loop-iteration stride in byte w.r.t. the left tensor
    std::vector< int64_t > m_loop_strides_left;
    //! per-loop-iteration stride in byte w.r.t. the right tensor
    std::vector< int64_t > m_loop_strides_right;
    //! per-loop-iteration stride in byte w.r.t. the auxiliary output tensor
    std::vector< int64_t > m_loop_strides_out_aux;
    //! per-loop-iteration stride in byte w.r.t. the output tensor
    std::vector< int64_t > m_loop_strides_out;

    //! true if the contraction loop interface was compiled
    bool m_compiled = false;

  protected:
    //! type of the first touch kernel
    kernel_t m_ktype_first_touch = UNDEFINED_KTYPE;
    //! type of the main kernel
    kernel_t m_ktype_main = UNDEFINED_KTYPE;
    //! type of the last touch kernel
    kernel_t m_ktype_last_touch = UNDEFINED_KTYPE;

    //! true if the outermost C dimension represents the complex dimension
    bool m_cpx_outer_c = false;

    //! complex stride of the left tensor
    int64_t m_cpx_stride_in_left_bytes = 0;
    //! complex stride of the right tensor
    int64_t m_cpx_stride_in_right_bytes = 0;
    //! complex stride of the auxiliary output tensor
    int64_t m_cpx_stride_out_aux_bytes = 0;
    //! complex stride of the output tensor
    int64_t m_cpx_stride_out_bytes = 0;

  public:
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
     * Kernel to pack the left input tensor of the main kernel.
     *
     * @param i_in  pointer to a data section of the input tensor.
     * @param i_out  pointer to output of packing.
     **/
    virtual void kernel_pack_left( void * i_in,
                                   void * io_out ) = 0;
    
    /**
     * Kernel to pack the right input tensor of the main kernel.
     *
     * @param i_in  pointer to a data section of the input tensor.
     * @param i_out  pointer to output of packing.
     **/
    virtual void kernel_pack_right( void * i_in,
                                    void * io_out ) = 0;

    /**
     * Initializes the the class.
     *
     * Shortcuts:
     *   C: batch dimensions which appears in all tensors.
     *   M: dimensions appear in left input and output.
     *   N: dimensions appear in right input and output.
     *   K: reduction dimensions which appear in both inputs,
     *
     * @param i_num_dims_c number of C dimensions.
     * @param i_num_dims_m number of M dimensions.
     * @param i_num_dims_n number of N dimensions.
     * @param i_num_dims_k number of K dimensions.
     * @param i_sizes_c sizes of the C dimensions.
     * @param i_sizes_m sizes of the M dimensions.
     * @param i_sizes_n sizes of the N dimensions.
     * @param i_sizes_k sizes of the K dimensions.
     * @param i_strides_in_left_c C strides of the left input tensor.
     * @param i_strides_in_left_m M strides of the left input tensor.
     * @param i_strides_in_left_k K strides of the left input tensor.
     * @param i_strides_in_right_c C strides of the right input tensor.
     * @param i_strides_in_right_n N strides of the right input tensor.
     * @param i_strides_in_right_k K strides of the right input tensor.
     * @param i_strides_out_aux_c C strides of the auxiliary output tensor.
     * @param i_strides_out_aux_m M strides of the auxiliary output tensor.
     * @param i_strides_out_aux_n N strides of the auxiliary output tensor.
     * @param i_strides_out_c C strides of the output tensor.
     * @param i_strides_out_m M strides of the output tensor.
     * @param i_strides_out_n N strides of the output tensor.
     * @param i_num_bytes_scalar_left number of bytes per scalar in the left tensor.
     * @param i_num_bytes_scalar_right number of bytes per scalar in the right tensor.
     * @param i_num_bytes_scalar_out number of bytes per scalar in the output tensor.
     * @param i_ktype_first_touch type of the first touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     **/
    void init( int64_t         i_num_dims_c,
               int64_t         i_num_dims_m,
               int64_t         i_num_dims_n,
               int64_t         i_num_dims_k,
               int64_t const * i_sizes_c,
               int64_t const * i_sizes_m,
               int64_t const * i_sizes_n,
               int64_t const * i_sizes_k,
               int64_t const * i_strides_in_left_c,
               int64_t const * i_strides_in_left_m,
               int64_t const * i_strides_in_left_k,
               int64_t const * i_strides_in_right_c,
               int64_t const * i_strides_in_right_n,
               int64_t const * i_strides_in_right_k,
               int64_t const * i_strides_out_aux_c,
               int64_t const * i_strides_out_aux_m,
               int64_t const * i_strides_out_aux_n,
               int64_t const * i_strides_out_c,
               int64_t const * i_strides_out_m,
               int64_t const * i_strides_out_n,
               int64_t         i_num_bytes_scalar_left,
               int64_t         i_num_bytes_scalar_right,
               int64_t         i_num_bytes_scalar_out,
               MemoryManager * i_memory,
               int64_t         i_size_packing_left,
               int64_t         i_size_packing_right,
               kernel_t        i_ktype_first_touch,
               kernel_t        i_ktype_main,
               kernel_t        i_ktype_last_touch );

    /**
     * Compiles the contraction loop interface.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile();

    /**
     * Derives the threading data for the contraction loops.
     * Parallelizes all loops such that the targeted number of tasks is reached or
     * all parallelizable loop dimensions have been exhausted.
     *
     * @param i_num_tasks number of tasks.
     **/
    err_t threading( int64_t i_num_tasks );

    /**
     * General purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param i_id_task task id which is executing the loop.
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_left pointer to the left tensor's data.
     * @param i_ptr_right pointer to the right tensor's data.
     * @param i_ptr_out_aux pointer to the auxiliary output tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     **/
    void contract_iter( int64_t         i_id_task,
                        int64_t         i_id_loop,
                        void    const * i_ptr_left,
                        void    const * i_ptr_right,
                        void    const * i_ptr_out_aux,
                        void          * i_ptr_out );

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
};

#endif