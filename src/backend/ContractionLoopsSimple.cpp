#include "ContractionLoopsSimple.h"

void einsum_ir::backend::ContractionLoopsSimple::init( int64_t         i_num_dims_c,
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
                                                       int64_t         i_num_bytes_scalar_out,
                                                       void         (* i_kernel_first_touch)( void * ),
                                                       void         (* i_kernel_inner)( void const *,
                                                                                        void const *,
                                                                                        void       * ),
                                                       void         (* i_kernel_last_touch)( void * ) ) {
  einsum_ir::backend::ContractionLoops::init( i_num_dims_c,
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
                                              i_strides_out_c,
                                              i_strides_out_m,
                                              i_strides_out_n,
                                              i_num_bytes_scalar_left,
                                              i_num_bytes_scalar_right,
                                              i_num_bytes_scalar_out );

  m_kernel_first_touch = i_kernel_first_touch;
  m_kernel_inner = i_kernel_inner;
  m_kernel_last_touch = i_kernel_last_touch;
}

void einsum_ir::backend::ContractionLoopsSimple::kernel_first_touch( void * io_out ) {
  if( m_kernel_first_touch != nullptr ) {
    m_kernel_first_touch( io_out );
  }
}

void einsum_ir::backend::ContractionLoopsSimple::kernel_inner( void const * i_left,
                                                               void const * i_right,
                                                               void       * io_out ) {
  m_kernel_inner( i_left,
                  i_right,
                  io_out );
}

void einsum_ir::backend::ContractionLoopsSimple::kernel_last_touch( void * io_out ) {
  if( m_kernel_last_touch != nullptr ) {
    m_kernel_last_touch( io_out );
  }
}