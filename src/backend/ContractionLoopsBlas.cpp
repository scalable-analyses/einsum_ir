#include "ContractionLoopsBlas.h"
#ifdef PP_EINSUM_IR_HAS_BLAS_NVPL
#include <nvpl_blas_cblas.h>
#else
#include <cblas.h>
#endif

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

void einsum_ir::backend::ContractionLoopsBlas::kernel_trans_32( int64_t   i_m,
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

void einsum_ir::backend::ContractionLoopsBlas::kernel_trans_64( int64_t   i_m,
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

void einsum_ir::backend::ContractionLoopsBlas::kernel_gemm_fp32( float         i_alpha,
                                                                 void  const * i_a,
                                                                 void  const * i_b,
                                                                 void        * io_c ) {
  cblas_sgemm( CblasColMajor,
               m_blas_trans_a ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_trans_b ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_size_m,
               m_blas_size_n,
               m_blas_size_k,
               i_alpha,
               (const float *) i_a,
               m_blas_ld_a,
               (const float *) i_b,
               m_blas_ld_b,
               1.0f,
               (float *) io_c,
               m_blas_ld_c );
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_gemm_fp64( double         i_alpha,
                                                                 void   const * i_a,
                                                                 void   const * i_b,
                                                                 void         * io_c ) {
  cblas_dgemm( CblasColMajor,
               m_blas_trans_a ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_trans_b ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans,
               m_blas_size_m,
               m_blas_size_n,
               m_blas_size_k,
               i_alpha,
               (const double *) i_a,
               m_blas_ld_a,
               (const double *) i_b,
               m_blas_ld_b,
               1.0,
               (double *) io_c,
               m_blas_ld_c );
}

void einsum_ir::backend::ContractionLoopsBlas::init( std::map< int64_t, int64_t > const * i_sizes,
                                                     std::map< int64_t, int64_t > const * i_strides_left,
                                                     std::map< int64_t, int64_t > const * i_strides_right,
                                                     std::map< int64_t, int64_t > const * i_strides_out_aux,
                                                     std::map< int64_t, int64_t > const * i_strides_out,
                                                     std::map< int64_t, dim_t >   const * i_dim_type,
                                                     std::vector<int64_t>               * i_loop_ids,
                                                     data_t                               i_blas_dtype,
                                                     bool                                 i_blas_trans_a,
                                                     bool                                 i_blas_trans_b,
                                                     int64_t                              i_blas_size_c,
                                                     int64_t                              i_blas_size_m,
                                                     int64_t                              i_blas_size_n,
                                                     int64_t                              i_blas_size_k,
                                                     int64_t                              i_blas_ld_a,
                                                     int64_t                              i_blas_ld_b,
                                                     int64_t                              i_blas_ld_c,
                                                     kernel_t                             i_ktype_first_touch,
                                                     kernel_t                             i_ktype_main,
                                                     kernel_t                             i_ktype_last_touch ) {

  m_num_bytes_scalar = ce_n_bytes( i_blas_dtype );

  ContractionLoops::init( i_sizes,
                          i_strides_left,
                          i_strides_right,
                          i_strides_out_aux,
                          i_strides_out,
                          i_dim_type,
                          i_loop_ids,
                          m_num_bytes_scalar,
                          m_num_bytes_scalar,
                          m_num_bytes_scalar,
                          i_ktype_first_touch,
                          i_ktype_main,
                          i_ktype_last_touch,
                          nullptr );

  m_blas_dtype   = i_blas_dtype;
  m_blas_trans_a = i_blas_trans_a;
  m_blas_trans_b = i_blas_trans_b;
  m_blas_size_c  = i_blas_size_c;
  m_blas_size_m  = i_blas_size_m;
  m_blas_size_n  = i_blas_size_n;
  m_blas_size_k  = i_blas_size_k;
  m_blas_ld_a    = i_blas_ld_a;
  m_blas_ld_b    = i_blas_ld_b;
  m_blas_ld_c    = i_blas_ld_c;
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_first_touch_part( void * io_out ) {
  if(    m_ktype_first_touch == kernel_t::ZERO
      || m_ktype_first_touch == kernel_t::CPX_ZERO ) {
    if( m_blas_dtype == data_t::FP32 ) {
      kernel_zero_32( m_blas_size_m * m_blas_size_c,
                      m_blas_size_n,
                      m_blas_ld_c,
                      io_out );
    }
    else if( m_blas_dtype == data_t::FP64 ) {
      kernel_zero_64( m_blas_size_m * m_blas_size_c,
                      m_blas_size_n,
                      m_blas_ld_c,
                      io_out );
    }
  }
  // packed GEMM primitive without zeroing: n[...]mc -> n[...]cm
  else if(    ( m_ktype_main == kernel_t::MADD || m_ktype_main == kernel_t::CPX_MADD )
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
      else {
        kernel_trans_64( m_blas_size_c,
                         m_blas_size_m,
                         m_blas_size_c,
                         m_blas_size_m,
                         l_out );
      }
    }
  }
}
void einsum_ir::backend::ContractionLoopsBlas::kernel_first_touch( void const *,
                                                                   void       * io_out ) {

  kernel_first_touch_part( io_out );
  if( m_cpx_outer_c ) {
    kernel_first_touch_part( (char *) io_out + m_cpx_stride_out_bytes );
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
    for( int64_t l_c = 0; l_c < m_blas_size_c; l_c++ ) {
      // derive pointers for current GEMM
      void const * l_left  = (char *) i_left  + l_c * m_blas_size_k * m_blas_ld_a * m_num_bytes_scalar;
      void const * l_right = (char *) i_right + l_c * m_blas_size_n * m_blas_ld_b * m_num_bytes_scalar;
      void       * l_out   = (char *) io_out  + l_c * m_blas_size_m * m_num_bytes_scalar;

      // execute GEMM
      if( m_blas_dtype == data_t::FP32 ) {
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
      else if( m_blas_dtype == data_t::FP64  ) {
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

void einsum_ir::backend::ContractionLoopsBlas::kernel_last_touch_part( void * io_out ) {

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
      else {
        kernel_trans_64( m_blas_size_m,
                         m_blas_size_c,
                         m_blas_size_m,
                         m_blas_size_c,
                         l_out );
      }
    }
  }
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_last_touch( void const *,
                                                                  void       * io_out ) {
  kernel_last_touch_part( io_out );
  if( m_cpx_outer_c ) {
    kernel_last_touch_part( (char *) io_out + m_cpx_stride_out_bytes );
  }
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_pack_left( void * i_in,
                                                                 void * io_out ) {                                                             
}

void einsum_ir::backend::ContractionLoopsBlas::kernel_pack_right( void * i_in,
                                                                  void * io_out ) {                                                        
}