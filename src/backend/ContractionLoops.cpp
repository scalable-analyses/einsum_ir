#include "ContractionLoops.h"

void einsum_ir::backend::ContractionLoops::init( int64_t         i_num_dims_c,
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
                                                 int64_t         i_num_bytes_scalar_out ) {
  m_num_dims_c = i_num_dims_c;
  m_num_dims_m = i_num_dims_m;
  m_num_dims_n = i_num_dims_n;
  m_num_dims_k = i_num_dims_k;

  m_sizes_c = i_sizes_c;
  m_sizes_m = i_sizes_m;
  m_sizes_n = i_sizes_n;
  m_sizes_k = i_sizes_k;

  m_strides_in_left_c = i_strides_in_left_c;
  m_strides_in_left_m = i_strides_in_left_m;
  m_strides_in_left_k = i_strides_in_left_k;

  m_strides_in_right_c = i_strides_in_right_c;
  m_strides_in_right_n = i_strides_in_right_n;
  m_strides_in_right_k = i_strides_in_right_k;

  m_strides_out_c = i_strides_out_c;
  m_strides_out_m = i_strides_out_m;
  m_strides_out_n = i_strides_out_n;

  m_num_bytes_scalar_left = i_num_bytes_scalar_left;
  m_num_bytes_scalar_right = i_num_bytes_scalar_right;
  m_num_bytes_scalar_out = i_num_bytes_scalar_out;
}

void einsum_ir::backend::ContractionLoops::contract_cnmk( char            i_dim_type,
                                                          int64_t         i_dim_count,
                                                          void    const * i_ptr_in_left,
                                                          void    const * i_ptr_in_right,
                                                          void          * i_ptr_out ) {
  int64_t l_num_dims = 0;
  int64_t const * l_sizes = nullptr;
  int64_t const * l_strides_in_left = nullptr;
  int64_t const * l_strides_in_right = nullptr;
  int64_t const * l_strides_out = nullptr;

  if( i_dim_type == 0 ) {
    l_num_dims = m_num_dims_c;
    l_sizes = m_sizes_c;
    l_strides_in_left = m_strides_in_left_c;
    l_strides_in_right = m_strides_in_right_c;
    l_strides_out = m_strides_out_c;
  }
  else if ( i_dim_type == 1 ) {
    l_num_dims = m_num_dims_n;
    l_sizes = m_sizes_n;
    l_strides_in_right = m_strides_in_right_n;
    l_strides_out = m_strides_out_n;
  }
  else if ( i_dim_type == 2 ) {
    l_num_dims = m_num_dims_m;
    l_sizes = m_sizes_m;
    l_strides_in_left = m_strides_in_left_m;
    l_strides_out = m_strides_out_m;
  }
  else if ( i_dim_type == 3 ) {
    l_num_dims = m_num_dims_k;
    l_sizes = m_sizes_k;
    l_strides_in_left = m_strides_in_left_k;
    l_strides_in_right = m_strides_in_right_k;
  }

  // first time a K dimension is observed:
  // execute first touch kernel
  if(    i_dim_count == 0
      && i_dim_type == 3 ) {
    kernel_first_touch( i_ptr_out );
  }

  // first case:
  //   current dimension type is not exhausted
  //   take care of additional dimension
  if( i_dim_count < l_num_dims ) {
    char * l_ptr_in_left = (char *) i_ptr_in_left;
    char * l_ptr_in_right = (char *) i_ptr_in_right;
    char * l_ptr_out = (char *) i_ptr_out;

    for( int64_t l_it = 0; l_it < l_sizes[i_dim_count]; l_it++ ) {
      contract_cnmk( i_dim_type,
                     i_dim_count+1,
                     l_ptr_in_left,
                     l_ptr_in_right,
                     l_ptr_out );

      if( i_dim_type != 1 ) {
        l_ptr_in_left += l_strides_in_left[i_dim_count] * m_num_bytes_scalar_left;
      }
      if( i_dim_type != 2 ) {
        l_ptr_in_right += l_strides_in_right[i_dim_count] * m_num_bytes_scalar_right;
      }
      if( i_dim_type != 3 ) {
        l_ptr_out += l_strides_out[i_dim_count] * m_num_bytes_scalar_out;
      }
    }
  }
  // second case:
  //   current dimension type is exhausted
  //   if available: continue with nested loops of next dim type
  else if ( i_dim_type < 3 ) {
    contract_cnmk( i_dim_type+1,
                   0,
                   i_ptr_in_left,
                   i_ptr_in_right,
                   i_ptr_out );
  }
  // third case:
  //   current dimension type is exhausted
  //   inside of the innermost nested loop
  else {
    // execute inner kernel
    kernel_inner( i_ptr_in_left,
                  i_ptr_in_right,
                  i_ptr_out );
  }

  // last K dimension finished all iterations:
  // execute last touch kernel
  if(    i_dim_count == l_num_dims-1
      && i_dim_type == 3 ) {
    kernel_last_touch( i_ptr_out );
  }
}

void einsum_ir::backend::ContractionLoops::contract( void const * i_tensor_in_left,
                                                     void const * i_tensor_in_right,
                                                     void       * io_tensor_out ) {
  contract_cnmk( 0,
                 0,
                 i_tensor_in_left,
                 i_tensor_in_right,
                 io_tensor_out );
}