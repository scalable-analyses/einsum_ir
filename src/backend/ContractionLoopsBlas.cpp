#include "ContractionLoopsBlas.h"
#include <cblas.h>

void einsum_ir::backend::ContractionLoopsBlas::kernel_zero_32( int64_t   i_m,
                                                               int64_t   i_n,
                                                               int64_t   i_ld,
                                                               void    * io_out ) {
  float * l_out = (float *) io_out;

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
#ifdef _OPENMP
#pragma omp simd
#endif
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      l_out[ l_n * i_ld + l_m ] = 0;
    }
  }
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_zero_64( int64_t   i_m,
                                                               int64_t   i_n,
                                                               int64_t   i_ld,
                                                               void    * io_out ) {
  double * l_out = (double *) io_out;

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
#ifdef _OPENMP
#pragma omp simd
#endif
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      l_out[ l_n * i_ld + l_m ] = 0;
    }
  }
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_zero_128( int64_t   i_m,
                                                                int64_t   i_n,
                                                                int64_t   i_ld,
                                                                void    * io_out ) {
  double * l_out = (double *) io_out;

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
#ifdef _OPENMP
#pragma omp simd
#endif
    for( int64_t l_m = 0; l_m < i_m*2; l_m++ ) {
      l_out[ l_n * i_ld * 2 + l_m ] = 0;
    }
  }
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_trans_32( int64_t   i_m,
                                                                int64_t   i_n,
                                                                int64_t   i_ld_a,
                                                                int64_t   i_ld_b,
                                                                void    * io_out ) {
  float * l_out = (float *) io_out;

  cblas_simatcopy( CblasColMajor,
                   CblasTrans,
                   i_m,
                   i_n,
                   1.0,
                   l_out,
                   i_ld_a,
                   i_ld_b );
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_trans_64( int64_t   i_m,
                                                                int64_t   i_n,
                                                                int64_t   i_ld_a,
                                                                int64_t   i_ld_b,
                                                                void    * io_out ) {
  double * l_out = (double *) io_out;

  cblas_dimatcopy( CblasColMajor,
                   CblasTrans,
                   i_m,
                   i_n,
                   1.0,
                   l_out,
                   i_ld_a,
                   i_ld_b );
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_trans_128( int64_t   i_m,
                                                                 int64_t   i_n,
                                                                 int64_t   i_ld_a,
                                                                 int64_t   i_ld_b,
                                                                 void    * io_out ) {
  double * l_out = (double *) io_out;
  double l_alpha[2] = { 1.0, 0.0 };

  cblas_zimatcopy( CblasColMajor,
                   CblasTrans,
                   i_m,
                   i_n,
                   l_alpha,
                   l_out,
                   i_ld_a,
                   i_ld_b );
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_gemm_fp32( void const * i_a,
                                                                 void const * i_b,
                                                                 void       * io_c ) {
  cblas_sgemm( m_blas_row_major ? CblasRowMajor : CblasColMajor,
               m_blas_trans_a ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_trans_b ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_size_m,
               m_blas_size_n,
               m_blas_size_k,
               m_blas_alpha,
               (const float *) i_a,
               m_blas_ld_a,
               (const float *) i_b,
               m_blas_ld_b,
               m_blas_beta,
               (float *) io_c,
               m_blas_ld_c );
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_gemm_fp64( void const * i_a,
                                                                 void const * i_b,
                                                                 void       * io_c ) {
  cblas_dgemm( m_blas_row_major ? CblasRowMajor : CblasColMajor,
               m_blas_trans_a ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_trans_b ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_size_m,
               m_blas_size_n,
               m_blas_size_k,
               m_blas_alpha,
               (const double *) i_a,
               m_blas_ld_a,
               (const double *) i_b,
               m_blas_ld_b,
               m_blas_beta,
               (double *) io_c,
               m_blas_ld_c );
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_gemm_cfp32( void const * i_a,
                                                                  void const * i_b,
                                                                  void       * io_c ) {
  float l_blas_alpha[2] = { 0 };
  l_blas_alpha[0] = m_blas_alpha;
  float l_blas_beta[2]  = { 0 };
  l_blas_beta[0] = m_blas_beta;

  cblas_cgemm( m_blas_row_major ? CblasRowMajor : CblasColMajor,
               m_blas_trans_a ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_trans_b ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_size_m,
               m_blas_size_n,
               m_blas_size_k,
               l_blas_alpha,
               (const float *) i_a,
               m_blas_ld_a,
               (const float *) i_b,
               m_blas_ld_b,
               l_blas_beta,
               (float *) io_c,
               m_blas_ld_c );
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_gemm_cfp64( void const * i_a,
                                                                  void const * i_b,
                                                                  void       * io_c ) {
  double l_blas_alpha[2] = { m_blas_alpha, 0.0 };
  double l_blas_beta[2]  = { m_blas_beta,  0.0  };
  cblas_zgemm( m_blas_row_major ? CblasRowMajor : CblasColMajor,
               m_blas_trans_a ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_trans_b ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_size_m,
               m_blas_size_n,
               m_blas_size_k,
               l_blas_alpha,
               (const double *) i_a,
               m_blas_ld_a,
               (const double *) i_b,
               m_blas_ld_b,
               l_blas_beta,
               (double *) io_c,
               m_blas_ld_c );
}

void einsum_ir::backend::ContractionLoopsBlas::init( int64_t                              i_num_dims_c,
                                                     int64_t                              i_num_dims_m,
                                                     int64_t                              i_num_dims_n,
                                                     int64_t                              i_num_dims_k,
                                                     int64_t                      const * i_sizes_c,
                                                     int64_t                      const * i_sizes_m,
                                                     int64_t                      const * i_sizes_n,
                                                     int64_t                      const * i_sizes_k,
                                                     int64_t                      const * i_strides_in_left_c,
                                                     int64_t                      const * i_strides_in_left_m,
                                                     int64_t                      const * i_strides_in_left_k,
                                                     int64_t                      const * i_strides_in_right_c,
                                                     int64_t                      const * i_strides_in_right_n,
                                                     int64_t                      const * i_strides_in_right_k,
                                                     int64_t                      const * i_strides_out_aux_c,
                                                     int64_t                      const * i_strides_out_aux_m,
                                                     int64_t                      const * i_strides_out_aux_n,
                                                     int64_t                      const * i_strides_out_c,
                                                     int64_t                      const * i_strides_out_m,
                                                     int64_t                      const * i_strides_out_n,
                                                     data_t                               i_blas_dtype,
                                                     bool                                 i_blas_row_major,
                                                     bool                                 i_blas_trans_a,
                                                     bool                                 i_blas_trans_b,
                                                     int64_t                              i_blas_size_c,
                                                     int64_t                              i_blas_size_m,
                                                     int64_t                              i_blas_size_n,
                                                     int64_t                              i_blas_size_k,
                                                     int64_t                              i_blas_ld_a,
                                                     int64_t                              i_blas_ld_b,
                                                     int64_t                              i_blas_ld_c,
                                                     double                               i_blas_alpha,
                                                     double                               i_blas_beta,
                                                     kernel_t                             i_ktype_first_touch,
                                                     kernel_t                             i_ktype_last_touch ) {

  m_num_bytes_scalar = ce_n_bytes( i_blas_dtype );

  ContractionLoops::init( i_num_dims_c,
                          i_num_dims_m,
                          i_num_dims_n,
                          i_num_dims_k,
                          i_sizes_c,
                          i_sizes_m,
                          i_sizes_n,
                          i_sizes_k,
                          i_strides_in_left_c,
                          i_strides_in_left_m,
                          i_strides_in_left_k,
                          i_strides_in_right_c,
                          i_strides_in_right_n,
                          i_strides_in_right_k,
                          i_strides_out_aux_c,
                          i_strides_out_aux_m,
                          i_strides_out_aux_n,
                          i_strides_out_c,
                          i_strides_out_m,
                          i_strides_out_n,
                          m_num_bytes_scalar,
                          m_num_bytes_scalar,
                          m_num_bytes_scalar );

  m_blas_dtype     = i_blas_dtype;
  m_blas_row_major = i_blas_row_major;
  m_blas_trans_a   = i_blas_trans_a;
  m_blas_trans_b   = i_blas_trans_b;
  m_blas_size_c    = i_blas_size_c;
  m_blas_size_m    = i_blas_size_m;
  m_blas_size_n    = i_blas_size_n;
  m_blas_size_k    = i_blas_size_k;
  m_blas_ld_a      = i_blas_ld_a;
  m_blas_ld_b      = i_blas_ld_b;
  m_blas_ld_c      = i_blas_ld_c;
  m_blas_alpha     = i_blas_alpha;
  m_blas_beta      = i_blas_beta;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_last_touch  = i_ktype_last_touch;
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_first_touch( void const *,
                                                                   void       * io_out ) {
  if( m_ktype_first_touch == kernel_t::ZERO ) {
    if( m_blas_dtype == data_t::FP32 ) {
      kernel_zero_32( m_blas_size_m * m_blas_size_c,
                      m_blas_size_n,
                      m_blas_ld_c,
                      io_out );
    }
    else if(    m_blas_dtype == data_t::FP64
             || m_blas_dtype == data_t::CFP32 ) {
      kernel_zero_64( m_blas_size_m * m_blas_size_c,
                      m_blas_size_n,
                      m_blas_ld_c,
                      io_out );
    }
    else {
      kernel_zero_128( m_blas_size_m * m_blas_size_c,
                       m_blas_size_n,
                       m_blas_ld_c,
                       io_out );
    }
  }
  // packed GEMM primitive without zeroing: n[...]mc -> n[...]cm
  else if(    m_ktype_first_touch == kernel_t::UNDEFINED_KTYPE
           && m_blas_size_c > 1 ) {
    for( int64_t l_n = 0; l_n < m_blas_size_n; l_n++ ) {
      void * l_out = (char *) io_out + l_n * m_blas_ld_c * m_num_bytes_scalar;

      if( m_blas_dtype == data_t::FP32 ) {
        kernel_trans_32( m_blas_size_c,
                         m_blas_size_m,
                         m_blas_size_c,
                         m_blas_size_m,
                         l_out );
      }
      else if(    m_blas_dtype == data_t::FP64
               || m_blas_dtype == data_t::CFP32 ) {
        kernel_trans_64( m_blas_size_c,
                         m_blas_size_m,
                         m_blas_size_c,
                         m_blas_size_m,
                         l_out );
      }
      else {
        kernel_trans_128( m_blas_size_c,
                          m_blas_size_m,
                          m_blas_size_c,
                          m_blas_size_m,
                          l_out );
      }
    }
  }
}

einsum_ir::err_t einsum_ir::backend::ContractionLoopsBlas::threading( int64_t i_num_tasks ) {
  err_t l_err = ContractionLoops::threading( i_num_tasks );
  if( l_err != err_t::SUCCESS ) {
    return l_err;
  }

  // disable threading in OpenBLAS
#ifdef OPENBLAS_VERSION
  openblas_set_num_threads( 1 );
#endif

  return err_t::SUCCESS;
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_main( void const * i_left,
                                                            void const * i_right,
                                                            void       * io_out ) {
  // GEMM primitive
  if( m_blas_size_c == 1 ) {
    if( m_blas_dtype == data_t::FP32 ) {
      kernel_gemm_fp32( i_left,
                        i_right,
                        io_out );
    }
    else if( m_blas_dtype == data_t::FP64  ) {
      kernel_gemm_fp64( i_left,
                        i_right,
                        io_out );
    }
    else if( m_blas_dtype == data_t::CFP32  ) {
      kernel_gemm_cfp32( i_left,
                         i_right,
                         io_out );
    }
    else {
      kernel_gemm_cfp64( i_left,
                         i_right,
                         io_out );
    }
  }
  // packed GEMM primitive: ckm, cnk -> n[...]cm
  else {
    for( int64_t l_c = 0; l_c < m_blas_size_c; l_c++ ) {
      // derive pointers for current GEMM
      void const * l_left  = (char *) i_left  + l_c * m_blas_size_k * m_blas_ld_a * m_num_bytes_scalar;
      void const * l_right = (char *) i_right + l_c * m_blas_size_n * m_blas_ld_b * m_num_bytes_scalar;
      void       * l_out   = (char *) io_out  + l_c * m_blas_size_m * m_num_bytes_scalar;

      // execute GEMM
      if( m_blas_dtype == data_t::FP32 ) {
      kernel_gemm_fp32( l_left,
                        l_right,
                        l_out );
      }
      else if( m_blas_dtype == data_t::FP64  ) {
      kernel_gemm_fp64( l_left,
                        l_right,
                        l_out );
      }
      else if( m_blas_dtype == data_t::CFP32 ) {
      kernel_gemm_cfp32( l_left,
                         l_right,
                         l_out );
      }
      else {
      kernel_gemm_cfp64( l_left,
                         l_right,
                         l_out );
      }
    }
  }
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_last_touch( void const *,
                                                                  void       * io_out ) {
  if( m_blas_size_c != 1 ) {
    // transpose part of the packed GEMM primitive: n[...]cm -> n[...]mc
    for( int64_t l_n = 0; l_n < m_blas_size_n; l_n++ ) {
      void * l_out = (char *) io_out + l_n * m_blas_ld_c * m_num_bytes_scalar;

      if( m_blas_dtype == data_t::FP32 ) {
        kernel_trans_32( m_blas_size_m,
                         m_blas_size_c,
                         m_blas_size_m,
                         m_blas_size_c,
                         l_out );
      }
      else if(    m_blas_dtype == data_t::FP64
               || m_blas_dtype == data_t::CFP32 ) {
        kernel_trans_64( m_blas_size_m,
                         m_blas_size_c,
                         m_blas_size_m,
                         m_blas_size_c,
                         l_out );
      }
      else {
        kernel_trans_128( m_blas_size_m,
                          m_blas_size_c,
                          m_blas_size_m,
                          m_blas_size_c,
                          l_out );
      }
    }
  }
}