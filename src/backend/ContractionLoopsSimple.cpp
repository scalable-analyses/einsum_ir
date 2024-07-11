#include "ContractionLoopsSimple.h"

void einsum_ir::backend::ContractionLoopsSimple::init( int64_t                              i_num_dims_c,
                                                       int64_t                              i_num_dims_m,
                                                       int64_t                              i_num_dims_n,
                                                       int64_t                              i_num_dims_k,
                                                       int64_t                      const * i_dim_ids_c,
                                                       int64_t                      const * i_dim_ids_m,
                                                       int64_t                      const * i_dim_ids_n,
                                                       int64_t                      const * i_dim_ids_k,
                                                       std::map< int64_t, int64_t > const * i_sizes,
                                                       std::map< int64_t, int64_t > const * i_strides_left,
                                                       std::map< int64_t, int64_t > const * i_strides_right,
                                                       std::map< int64_t, int64_t > const * i_strides_out_aux,
                                                       std::map< int64_t, int64_t > const * i_strides_out,
                                                       std::map< int64_t, dim_t >   const * i_dim_type,
                                                       int64_t                              i_num_bytes_scalar_left,
                                                       int64_t                              i_num_bytes_scalar_right,
                                                       int64_t                              i_num_bytes_scalar_out,
                                                       void                              (* i_kernel_first_touch)( void const *,
                                                                                                                   void       * ),
                                                       void                              (* i_kernel_main)( void const *,
                                                                                                            void const *,
                                                                                                            void       * ),
                                                       void                              (* i_kernel_last_touch)( void const *,
                                                                                                                  void       * ) ) {
  einsum_ir::backend::ContractionLoops::init( i_num_dims_c,
                                              i_num_dims_m,
                                              i_num_dims_n,
                                              i_num_dims_k,
                                              i_dim_ids_c,
                                              i_dim_ids_m,
                                              i_dim_ids_n,
                                              i_dim_ids_k,
                                              i_sizes,
                                              i_strides_left,
                                              i_strides_right,
                                              i_strides_out_aux,
                                              i_strides_out,
                                              i_dim_type,
                                              i_num_bytes_scalar_left,
                                              i_num_bytes_scalar_right,
                                              i_num_bytes_scalar_out,
                                              kernel_t::CUSTOM_KTYPE,
                                              kernel_t::CUSTOM_KTYPE,
                                              kernel_t::CUSTOM_KTYPE,
                                              nullptr );

  m_kernel_first_touch = i_kernel_first_touch;
  m_kernel_main = i_kernel_main;
  m_kernel_last_touch = i_kernel_last_touch;
}

void einsum_ir::backend::ContractionLoopsSimple::kernel_first_touch( void const * i_out_aux,
                                                                     void       * io_out ) {
  if( m_kernel_first_touch != nullptr ) {
    m_kernel_first_touch( i_out_aux,
                          io_out );
  }
}

void einsum_ir::backend::ContractionLoopsSimple::kernel_main( void const * i_left,
                                                              void const * i_right,
                                                              void       * io_out ) {
  m_kernel_main( i_left,
                 i_right,
                 io_out );
}

void einsum_ir::backend::ContractionLoopsSimple::kernel_last_touch( void const * i_out_aux,
                                                                    void       * io_out ) {
  if( m_kernel_last_touch != nullptr ) {
    m_kernel_last_touch( i_out_aux,
                         io_out );
  }
}