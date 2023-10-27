#include "UnaryLoopsSimple.h"

void einsum_ir::backend::UnaryLoopsSimple::init( int64_t         i_num_dims,
                                                 int64_t const * i_sizes,
                                                 int64_t const * i_strides_in,
                                                 int64_t const * i_strides_out,
                                                 int64_t         i_num_bytes_in,
                                                 int64_t         i_num_bytes_out,
                                                 void         (* i_kernel_main)( void const *,
                                                                                 void       * ) ) {
  einsum_ir::backend::UnaryLoops::init( i_num_dims,
                                        i_sizes,
                                        i_strides_in,
                                        i_strides_out,
                                        i_num_bytes_in,
                                        i_num_bytes_out );

  m_kernel_main = i_kernel_main;
}

void einsum_ir::backend::UnaryLoopsSimple::kernel_main( void const * i_ptr_in,
                                                        void       * io_ptr_out ) {
  m_kernel_main( i_ptr_in,
                 io_ptr_out );
}