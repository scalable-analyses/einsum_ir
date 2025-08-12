#ifndef ETOPS_BINARY_CONTRACTION_BACKEND_BLAS
#define ETOPS_BINARY_CONTRACTION_BACKEND_BLAS

#include "ContractionBackend.h"

namespace etops {
  namespace binary {
    class ContractionBackendBlas;
  }
}

class etops::binary::ContractionBackendBlas: public ContractionBackend {
  private:
    //! number of bytes in a scalar
    int64_t m_num_bytes_scalar = 0;

    //! true if the outermost C dimension represents the complex dimension
    bool m_cpx_outer_c = false;

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

    /**
     * Compiles all kernels
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile_kernels();
};

#endif
