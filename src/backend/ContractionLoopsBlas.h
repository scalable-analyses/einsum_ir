#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS_BLAS
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS_BLAS

#include "ContractionLoops.h"

namespace einsum_ir {
  namespace backend {
    class ContractionLoopsBlas;
  }
}

class einsum_ir::backend::ContractionLoopsBlas: public ContractionLoops {
  private:
    //! number of bytes in a scalar
    int64_t m_num_bytes_scalar = 0;

    //! BLAS data type
    data_t m_blas_dtype = UNDEFINED_DTYPE;
    //! A is transposed if true, not transposed otherwise
    bool m_blas_trans_a = false;
    //! B is transposed if true, not transposed otherwise
    bool m_blas_trans_b = false;

    //! size of the C dimension w.r.t. packed GEMMs
    int64_t m_blas_size_c = 0;
    //! size of the M dimension in the BLAS calls
    int64_t m_blas_size_m = 0;
    //! size of the N dimension in the BLAS calls
    int64_t m_blas_size_n = 0;
    //! size of the K dimension in the BLAS calls
    int64_t m_blas_size_k = 0;

    //! leading dimension of the BLAS matrix A (part of the left tensor)
    int64_t m_blas_ld_a = 0;
    //! leading dimension of the BLAS matrix B (part of the right tensor)
    int64_t m_blas_ld_b = 0;
    //! leading dimension of the BLAS matrix C (part of the output tensor)
    int64_t m_blas_ld_c = 0;

    /**
     * 32-bit kernel zeroing a column-major matrix.
     *
     * @param i_m number of rows.
     * @param i_n number of columns.
     * @param i_ld leading dimension.
     * @param io_out pointer to the matrix.
     */
    static void kernel_zero_32( int64_t   i_m,
                                int64_t   i_n,
                                int64_t   i_ld,
                                void    * io_out );

    /**
     * 64-bit kernel zeroing a column-major matrix.
     *
     * @param i_m number of rows.
     * @param i_n number of columns.
     * @param i_ld leading dimension.
     * @param io_out pointer to the matrix.
     */
    static void kernel_zero_64( int64_t   i_m,
                                int64_t   i_n,
                                int64_t   i_ld,
                                void    * io_out );

    /**
     * 32-bit kernel transposing a column-major matrix.
     * The matrix is transposed in-place.
     *
     * @param i_m number of rows.
     * @param i_n number of columns.
     * @param i_ld_a leading dimension of the input matrix.
     * @param i_ld_b leading dimension of the output matrix.
     * @param io_out pointer to the matrix. 
     **/
    static void kernel_trans_32( int64_t   i_m,
                                 int64_t   i_n,
                                 int64_t   i_ld_a,
                                 int64_t   i_ld_b,
                                 void    * io_out );
    /**
     * 64-bit kernel transposing a column-major matrix.
     * The matrix is transposed in-place.
     *
     * @param i_m number of rows.
     * @param i_n number of columns.
     * @param i_ld_a leading dimension of the input matrix.
     * @param i_ld_b leading dimension of the output matrix.
     * @param io_out pointer to the matrix. 
     **/
    static void kernel_trans_64( int64_t   i_m,
                                 int64_t   i_n,
                                 int64_t   i_ld_a,
                                 int64_t   i_ld_b,
                                 void    * io_out );

    /**
     * FP32 GEMM kernel.
     *
     * @param i_alpha parameter alpha.
     * @param i_a pointer to matrix A.
     * @param i_b pointer to matrix B.
     * @param io_c pointer to matrix C.
     **/
    void kernel_gemm_fp32( float         i_alpha,
                           void  const * i_a,
                           void  const * i_b,
                           void        * io_c );

    /**
     * FP64 GEMM kernel.
     *
     * @param i_alpha parameter alpha.
     * @param i_a pointer to matrix A.
     * @param i_b pointer to matrix B.
     * @param io_c pointer to matrix C.
     **/
    void kernel_gemm_fp64( double         i_alpha,
                           void   const * i_a,
                           void   const * i_b,
                           void         * io_c );

    /**
     * Partially executes the first touch kernel on the given real or imaginary data section of the tensor.
     *
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_first_touch_part( void * io_out );

    /**
     * Partially executes the last touch kernel on the given real or imaginary data section of the tensor.
     *
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_last_touch_part( void * io_out );

  public:
    /**
     * Initializes the BLAS-based contraction loops.
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
     * @param i_blas_dtype BLAS data type.
     * @param i_blas_trans_a A is transposed if true, not transposed otherwise.
     * @param i_blas_trans_b B is transposed if true, not transposed otherwise.
     * @param i_blas_size_c size of the C dimension w.r.t. packed GEMMs.
     * @param i_blas_size_m size of the M dimension in the BLAS calls.
     * @param i_blas_size_n size of the N dimension in the BLAS calls.
     * @param i_blas_size_k size of the K dimension in the BLAS calls.
     * @param i_blas_ld_a leading dimension of the BLAS matrix A (part of the left tensor).
     * @param i_blas_ld_b leading dimension of the BLAS matrix B (part of the right tensor).
     * @param i_blas_ld_c leading dimension of the BLAS matrix C (part of the output tensor).
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
               data_t          i_blas_dtype,
               bool            i_blas_trans_a,
               bool            i_blas_trans_b,
               int64_t         i_blas_size_c,
               int64_t         i_blas_size_m,
               int64_t         i_blas_size_n,
               int64_t         i_blas_size_k,
               int64_t         i_blas_ld_a,
               int64_t         i_blas_ld_b,
               int64_t         i_blas_ld_c,
               kernel_t        i_ktype_first_touch,
               kernel_t        i_ktype_main,
               kernel_t        i_ktype_last_touch );

    /**
     * Derives the threading data for the contraction loops.
     * Parallelizes all loops such that the targeted number of tasks is reached or
     * all parallelizable loop dimensions have been exhausted.
     *
     * @param i_num_tasks number of tasks.
     **/
    err_t threading( int64_t i_num_tasks );

    /**
     * Executes the first touch kernel on the given data section of the tensor.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_first_touch( void const * i_out_aux,
                             void       * io_out );

    /**
     * Executes the main kernel on the given data sections of the tensors.
     *
     * @param i_left pointer to a data section of the left tensor.
     * @param i_right pointer to a data section of the right tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_main( void const * i_left,
                      void const * i_right,
                      void       * io_out );

    /**
     * Executes the last touch kernel on the given data section of the tensor.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    void kernel_last_touch( void const * i_out_aux,
                            void       * io_out );
};

#endif
