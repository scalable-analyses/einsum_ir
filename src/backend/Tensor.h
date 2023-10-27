#ifndef EINSUM_IR_BACKEND_TENSOR
#define EINSUM_IR_BACKEND_TENSOR

#include <cstdint>
#include <map>
#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class Tensor;
  }
}

class einsum_ir::backend::Tensor {
  public:
    /**
     * Derives the size of the given tensor in bytes.
     *
     * @param i_bytes_per_entry number of bytes per entry in the tensor.
     * @param i_num_dims number of dimensions.
     * @param i_dim_ids dimension ids.
     * @param i_dim_sizes sizes of the dimension.
     **/
    static int64_t size( int64_t                              i_bytes_per_entry,
                         int64_t                              i_num_dims,
                         int64_t                      const * i_dim_ids,
                         std::map< int64_t, int64_t > const & i_dim_sizes );
};

#endif