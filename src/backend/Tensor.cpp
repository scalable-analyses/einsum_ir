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