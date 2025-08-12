#include "ContractionBackendBlas.h"
#ifdef PP_EINSUM_IR_HAS_BLAS_NVPL
#include <nvpl_blas_cblas.h>
#else
#include <cblas.h>
#endif

void etops::binary::ContractionBackendBlas::kernel_zero_32( int64_t   i_m,
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

void etops::binary::ContractionBackendBlas::kernel_zero_64( int64_t   i_m,
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

void etops::binary::ContractionBackendBlas::kernel_trans_32( int64_t   i_m,
                                                             int64_t   i_n,
                                                             int64_t   i_ld_a,
                                                             int64_t   i_ld_b,
                                                             void    * io_out ) {
  float * l_out = (float *) io_out;

#ifdef PP_EINSUM_IR_HAS_BLAS_IMATCOPY
  cblas_simatcopy( CblasColMajor,
                   CblasTrans,
                   i_m,
                   i_n,
                   1.0,
                   l_out,
                   i_ld_a,
                   i_ld_b );
#else
  // TODO: inefficient fall back implementation
  //       1) repeated memory allocations inside of OMP
  //       2) transpose + copy
  float * l_scratch = new float[ i_m * i_n ];

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      l_scratch[ l_n * i_m + l_m ] = l_out[ l_n * i_ld_a + l_m ];
    }
  }

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      l_out[ l_m * i_ld_b + l_n ] = l_scratch[ l_n * i_m + l_m ];
    }
  }

  delete [] l_scratch;
#endif
}

void etops::binary::ContractionBackendBlas::kernel_trans_64( int64_t   i_m,
                                                             int64_t   i_n,
                                                             int64_t   i_ld_a,
                                                             int64_t   i_ld_b,
                                                             void    * io_out ) {
    double * l_out = (double *) io_out;

#ifdef PP_EINSUM_IR_HAS_BLAS_IMATCOPY
  cblas_dimatcopy( CblasColMajor,
                   CblasTrans,
                   i_m,
                   i_n,
                   1.0,
                   l_out,
                   i_ld_a,
                   i_ld_b );
#else
  // TODO: inefficient fall back implementation
  //       1) repeated memory allocations inside of OMP
  //       2) transpose + copy
  double * l_scratch = new double[ i_m * i_n ];

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      l_scratch[ l_n * i_m + l_m ] = l_out[ l_n * i_ld_a + l_m ];
    }
  }

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      l_out[ l_m * i_ld_b + l_n ] = l_scratch[ l_n * i_m + l_m ];
    }
  }

  delete [] l_scratch;
#endif
}

void etops::binary::ContractionBackendBlas::kernel_gemm_fp32( float         i_alpha,
                                                              void  const * i_a,
                                                              void  const * i_b,
                                                              void        * io_c ) {
  cblas_sgemm( CblasColMajor,
               m_trans_a ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_trans_b ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_m,
               m_n,
               m_k,
               i_alpha,
               (const float *) i_a,
               m_lda,
               (const float *) i_b,
               m_ldb,
               1.0f,
               (float *) io_c,
               m_ldc );
}

void etops::binary::ContractionBackendBlas::kernel_gemm_fp64( double         i_alpha,
                                                              void   const * i_a,
                                                              void   const * i_b,
                                                              void         * io_c ) {
  cblas_dgemm( CblasColMajor,
               m_trans_a ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_trans_b ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_m,
               m_n,
               m_k,
               i_alpha,
               (const double *) i_a,
               m_lda,
               (const double *) i_b,
               m_ldb,
               1.0,
               (double *) io_c,
               m_ldc );
}

void etops::binary::ContractionBackendBlas::kernel_first_touch_part( void * io_out ) {
  if(    m_ktype_first_touch == kernel_t::ZERO
      || m_ktype_first_touch == kernel_t::CPX_ZERO ) {
    if( m_dtype_comp == data_t::FP32 ) {
      kernel_zero_32( m_m * m_r,
                      m_n,
                      m_ldc,
                      io_out );
    }
    else if( m_dtype_comp == data_t::FP64 ) {
      kernel_zero_64( m_m * m_r,
                      m_n,
                      m_ldc,
                      io_out );
    }
  }
  // packed GEMM primitive without zeroing: n[...]mc -> n[...]cm
  if(    ( m_ktype_main == kernel_t::PACKED_MADD || m_ktype_main == kernel_t::CPX_PACKED_MADD )
      && m_r > 1 ) {
    for( uint64_t l_n = 0; l_n < m_n; l_n++ ) {
      void * l_out = (char *) io_out + l_n * m_ldc * m_num_bytes_scalar;
      if( m_dtype_comp == data_t::FP32 ) {
        kernel_trans_32( m_r,
                         m_m,
                         m_r,
                         m_m,
                         l_out );
      }
      else {
        kernel_trans_64( m_r,
                         m_m,
                         m_r,
                         m_m,
                         l_out );
      }
    }
  }
}
void etops::binary::ContractionBackendBlas::kernel_first_touch( void const *,
                                                                void       * io_out ) {
  kernel_first_touch_part( io_out );
  if( m_cpx_outer_c ) {
    kernel_first_touch_part( (char *) io_out + m_cpx_stride_out_bytes );
  }
}
etops::err_t etops::binary::ContractionBackendBlas::compile_kernels(){
  m_num_bytes_scalar = ce_n_bytes( m_dtype_comp );

  m_cpx_outer_c = m_ktype_main == kernel_t::CPX_MADD || m_ktype_main == kernel_t::CPX_PACKED_MADD;

  if( m_ktype_main == kernel_t::PACKED_MADD || m_ktype_main == kernel_t::CPX_PACKED_MADD){
    if(m_ktype_first_touch == kernel_t::UNDEFINED_KTYPE ){
      m_ktype_first_touch = kernel_t::CPX_COPY;
    }
    m_ktype_last_touch = kernel_t::CPX_COPY;
  }

  // disable threading in OpenBLAS
#ifdef OPENBLAS_VERSION
  openblas_set_num_threads( 1 );
#endif

  return err_t::SUCCESS;
}


void etops::binary::ContractionBackendBlas::kernel_main( void const * i_left,
                                                         void const * i_right,
                                                         void       * io_out ) {
  // GEMM primitive
  if( m_r == 1 ) {
    if( m_dtype_comp == data_t::FP32 ) {
      kernel_gemm_fp32( 1.0f,
                        i_left,
                        i_right,
                        io_out );
      if( m_cpx_outer_c ) {
        // imag += real * imag
        kernel_gemm_fp32( 1.0f,
                          i_left,
                          (char *) i_right + m_cpx_stride_in_right_bytes,
                          (char *) io_out  + m_cpx_stride_out_bytes );
        // imag += imag * real
        kernel_gemm_fp32( 1.0f,
                          (char *) i_left  + m_cpx_stride_in_left_bytes,
                          i_right,
                          (char *) io_out  + m_cpx_stride_out_bytes );
        // real += imag * imag
        kernel_gemm_fp32( -1.0f,
                          (char *) i_left  + m_cpx_stride_in_left_bytes,
                          (char *) i_right + m_cpx_stride_in_right_bytes,
                          (char *) io_out  );
      }
    }
    else {
      kernel_gemm_fp64( 1.0,
                        i_left,
                        i_right,
                        io_out );
      if( m_cpx_outer_c ) {
        // imag += real * imag
        kernel_gemm_fp64( 1.0,
                          i_left,
                          (char *) i_right + m_cpx_stride_in_right_bytes,
                          (char *) io_out  + m_cpx_stride_out_bytes );
        // imag += imag * real
        kernel_gemm_fp64( 1.0,
                          (char *) i_left  + m_cpx_stride_in_left_bytes,
                          i_right,
                          (char *) io_out  + m_cpx_stride_out_bytes );
        // real += imag * imag
        kernel_gemm_fp64( -1.0,
                          (char *) i_left  + m_cpx_stride_in_left_bytes,
                          (char *) i_right + m_cpx_stride_in_right_bytes,
                          (char *) io_out  );
      }
    }
  }
  // packed GEMM primitive: ckm, cnk -> n[...]cm
  else {
    for( uint64_t l_c = 0; l_c < m_r; l_c++ ) {
      // derive pointers for current GEMM
      void const * l_left  = (char *) i_left  + l_c * m_packed_stride_a * m_num_bytes_scalar;
      void const * l_right = (char *) i_right + l_c * m_packed_stride_b * m_num_bytes_scalar;
      void       * l_out   = (char *) io_out  + l_c * m_m * m_num_bytes_scalar;
      // execute GEMM
      if( m_dtype_comp == data_t::FP32 ) {
        kernel_gemm_fp32( 1.0f,
                          l_left,
                          l_right,
                          l_out );
        if( m_cpx_outer_c ) {
          // imag += real * imag
          kernel_gemm_fp32( 1.0f,
                            l_left,
                            (char *) l_right + m_cpx_stride_in_right_bytes,
                            (char *) l_out   + m_cpx_stride_out_bytes );
          // imag += imag * real
          kernel_gemm_fp32( 1.0f,
                            (char *) l_left  + m_cpx_stride_in_left_bytes,
                            l_right,
                            (char *) l_out   + m_cpx_stride_out_bytes );
          // real += imag * imag
          kernel_gemm_fp32( -1.0f,
                            (char *) l_left  + m_cpx_stride_in_left_bytes,
                            (char *) l_right + m_cpx_stride_in_right_bytes,
                            (char *) l_out  );
        }
      }
      else if( m_dtype_comp == data_t::FP64  ) {
        kernel_gemm_fp64( 1.0f,
                          l_left,
                          l_right,
                          l_out );
        if( m_cpx_outer_c ) {
          // imag += real * imag
          kernel_gemm_fp64( 1.0,
                            l_left,
                            (char *) l_right + m_cpx_stride_in_right_bytes,
                            (char *) l_out   + m_cpx_stride_out_bytes );
          // imag += imag * real
          kernel_gemm_fp64( 1.0,
                            (char *) l_left  + m_cpx_stride_in_left_bytes,
                            l_right,
                            (char *) l_out   + m_cpx_stride_out_bytes );
          // real += imag * imag
          kernel_gemm_fp64( -1.0,
                            (char *) l_left  + m_cpx_stride_in_left_bytes,
                            (char *) l_right + m_cpx_stride_in_right_bytes,
                            (char *) l_out  );
        }
      }
    }
  }
}

void etops::binary::ContractionBackendBlas::kernel_last_touch_part( void * io_out ) {

  if( m_r != 1 ) {
    // transpose part of the packed GEMM primitive: n[...]cm -> n[...]mc
    for( uint64_t l_n = 0; l_n < m_n; l_n++ ) {
      void * l_out = (char *) io_out + l_n * m_ldc * m_num_bytes_scalar;

      if( m_dtype_comp == data_t::FP32 ) {
        kernel_trans_32( m_m,
                         m_r,
                         m_m,
                         m_r,
                         l_out );
      }
      else {
        kernel_trans_64( m_m,
                         m_r,
                         m_m,
                         m_r,
                         l_out );
      }
    }
  }
}

void etops::binary::ContractionBackendBlas::kernel_last_touch( void const *,
                                                               void       * io_out ) {
  kernel_last_touch_part( io_out );
  if( m_cpx_outer_c ) {
    kernel_last_touch_part( (char *) io_out + m_cpx_stride_out_bytes );
  }
}
