#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS

#include <cstdint>

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

    //! upper dimension type limit for spawning tasks
    char m_threading_limit_dim_type = -1;

    //! upper dimension count limit for spawning tasks
    int64_t m_threading_limit_dim_count = -1;

    //! grain size for the innermost loop
    int64_t m_threading_grain_size_inner_most = -1;

  public:
    /**
     * Kernel applied to the output tensor before the contraction.
     *
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_first_touch( void * io_out ) = 0;

    /**
     * Kernel applied to the output tensor after the contraction.
     *
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_last_touch( void * io_out ) = 0;

    /**
     * Kernel called in the innermost loop.
     *
     * @param i_left pointer to a data section of the left tensor.
     * @param i_right pointer to a data section of the right tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_inner( void const * i_left,
                               void const * i_right,
                               void       * io_out ) = 0;

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
     * @param i_strides_out_c C strides of the output tensor.
     * @param i_strides_out_m M strides of the output tensor.
     * @param i_strides_out_n N strides of the output tensor.
     * @param i_num_bytes_scalar_left number of bytes per scalar in the left tensor.
     * @param i_num_bytes_scalar_right number of bytes per scalar in the right tensor.
     * @param i_num_bytes_scalar_out number of bytes per scalar in the output tensor.
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
               int64_t const * i_strides_out_c,
               int64_t const * i_strides_out_m,
               int64_t const * i_strides_out_n,
               int64_t         i_num_bytes_scalar_left,
               int64_t         i_num_bytes_scalar_right,
               int64_t         i_num_bytes_scalar_out );

    /**
     * Sets the threading for the contraction loops.
     * Parallelizes all loops such that the targeted number of tasks is reached or
     * all parallelizable loop dimensions have been exhausted.
     *
     * @param i_num_tasks_target number of targeted tasks. 
     **/
    void threading( int64_t i_num_tasks_target );

    /**
     * Contracts the two input tensors.
     * Uses C-M-N-K (outer-to-inner dimensions) for the ordering.
     *
     * @param i_dim_type dimension type of the current recursion level: C (0), M (1), N (2) or K (3).
     * @param i_dim_count counter for the current dimension type.
     * @param i_ptr_in_left pointer to the left tensor.
     * @param i_ptr_in_right pointer to the right tensor.
     * @param i_ptr_out pointer to the output tensor.
     **/
    void contract_cnmk( char            i_dim_type,
                        int64_t         i_dim_count,
                        void    const * i_ptr_in_left,
                        void    const * i_ptr_in_right,
                        void          * i_ptr_out );

    /**
     * Contracts the two tensors.
     *
     * @param i_tensor_in_left left tensor.
     * @param i_tensor_in_right right tensor.
     * @param io_tensor_out output tensor.
     **/
    void contract( void const * i_tensor_in_left,
                   void const * i_tensor_in_right,
                   void       * io_tensor_out );
};

#endif