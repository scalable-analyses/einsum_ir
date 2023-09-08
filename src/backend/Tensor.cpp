#include "Tensor.h"
#include <vector>
#include <cassert>

int64_t einsum_ir::backend::Tensor::size( int64_t                              i_bytes_per_entry,
                                          int64_t                              i_num_dims,
                                          int64_t                      const * i_dim_ids,
                                          std::map< int64_t, int64_t > const & i_dim_sizes ) {
  int64_t l_size = i_bytes_per_entry;
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    int64_t l_dim_id = i_dim_ids[l_di];
    l_size *= i_dim_sizes.at( l_dim_id );
  }

  return l_size;
}

template< typename T_IN,
          typename T_OUT >
void einsum_ir::backend::Tensor::transpose( int64_t         i_num_dims,
                                            int64_t const * i_sizes,
                                            int64_t const * i_strides_in,
                                            int64_t const * i_strides_out,
                                            T_IN    const * i_data_ptr_in,
                                            T_OUT         * o_data_ptr_out ) {
  T_IN  const * l_tensor_in  = i_data_ptr_in;
  T_OUT       * l_tensor_out = o_data_ptr_out;

  if( i_num_dims > 0 ) {
    for( int64_t l_it = 0; l_it < i_sizes[0]; l_it++ ) {
      transpose( i_num_dims-1,
                 i_sizes+1,
                 i_strides_in+1,
                 i_strides_out+1,
                 l_tensor_in,
                 l_tensor_out );

      l_tensor_in += i_strides_in[0];
      l_tensor_out += i_strides_out[0];
    }
  }
  else {
    *l_tensor_out = *l_tensor_in;
  }
}

void einsum_ir::backend::Tensor::permute( int64_t         i_num_dims,
                                          int64_t const * i_dim_sizes,
                                          int64_t const * i_permutation,
                                          data_t          i_dtype_in,
                                          data_t          i_dtype_out,
                                          void    const * i_data_ptr_in,
                                          void          * o_data_ptr_out ) {
  std::vector< int64_t > l_sizes( i_num_dims );
  std::vector< int64_t > l_strides_in( i_num_dims );
  std::vector< int64_t > l_strides_out( i_num_dims );

  int64_t l_di = i_num_dims - 1;

  int64_t l_stride_in_tmp = 1;
  int64_t l_stride_out_tmp = 1;

  while( l_di >= 0 ) {
    int64_t l_id_out = i_permutation[l_di];

    l_strides_in[l_di]  = l_stride_in_tmp;
    l_strides_out[l_id_out] = l_stride_out_tmp;

    l_stride_in_tmp *= i_dim_sizes[l_di];
    l_stride_out_tmp *= i_dim_sizes[l_id_out];

    l_di--;
  }

  if(        i_dtype_in  == einsum_ir::FP32
          && i_dtype_out == einsum_ir::FP32 ) {
    float const * l_tensor_in = (float const *) i_data_ptr_in;
    float * l_tensor_out = (float *) o_data_ptr_out;

    transpose( i_num_dims,
               i_dim_sizes,
               l_strides_in.data(),
               l_strides_out.data(),
               l_tensor_in,
               l_tensor_out );
  }
  else if(    i_dtype_in  == einsum_ir::FP64
           && i_dtype_out == einsum_ir::FP64 ) {
    double const * l_tensor_in = (double const *) i_data_ptr_in;
    double * l_tensor_out = (double *) o_data_ptr_out;

    transpose( i_num_dims,
               i_dim_sizes,
               l_strides_in.data(),
               l_strides_out.data(),
               l_tensor_in,
               l_tensor_out );
  }
  else {
    assert( false );
  }
}

void einsum_ir::backend::Tensor::permute( int64_t                              i_num_dims,
                                          std::map< int64_t, int64_t > const & i_dim_sizes,
                                          int64_t                      const * i_dim_ids_in,
                                          int64_t                      const * i_dim_ids_out,
                                          data_t                               i_dtype_in,
                                          data_t                               i_dtype_out,
                                          void                         const * i_data_ptr_in,
                                          void                               * o_data_ptr_out ) {
  std::vector< int64_t > l_sizes( i_num_dims );
  std::vector< int64_t > l_permutation( i_num_dims );

  for( int64_t l_di_in = 0; l_di_in < i_num_dims; l_di_in++ ) {
    int64_t l_dim_id = i_dim_ids_in[l_di_in];

    l_sizes[l_di_in] = i_dim_sizes.at( l_dim_id );

    for( int64_t l_di_out = 0; l_di_out < i_num_dims; l_di_out++ ) {
      if( i_dim_ids_in[l_di_in] == i_dim_ids_out[l_di_out] ) {
        l_permutation[l_di_out] = l_di_in;
      }
    }
  }

  permute( i_num_dims,
           l_sizes.data(),
           l_permutation.data(),
           i_dtype_in,
           i_dtype_out,
           i_data_ptr_in,
           o_data_ptr_out );
}