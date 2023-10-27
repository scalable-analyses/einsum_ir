#ifndef EINSUM_IR_BACKEND_UNARY_TPP
#define EINSUM_IR_BACKEND_UNARY_TPP

#include <vector>
#include "Unary.h"
#include "UnaryLoopsTpp.h"

namespace einsum_ir {
  namespace backend {
    class UnaryTpp;
  }
}

class einsum_ir::backend::UnaryTpp: public Unary {
  private:
    //! unary loop interface
    UnaryLoopsTpp m_unary_loops;

    //! LIBXSMM-based unary main TPP
    libxsmm_meltwfunction_unary m_xmm_kernel_main = nullptr;

    //! loop sizes
    std::vector< int64_t > m_loop_sizes;

    //! input strides of the loops
    std::vector< int64_t > m_loop_strides_in;

    //! output strides of the loops
    std::vector< int64_t > m_loop_strides_out;

    /**
     * Converts the given native datatype to a LIBXSMM datatype.
     *
     * @param i_data_type native datatype.
     * @return corresponding LIBXSMM datatype.
     */
    static libxsmm_datatype dtype_to_libxsmm( data_t i_dtype );

  public:
    /**
     * Compiles the unary operation.
     *
     * @return SUCCESS if successful, error code otherwise.
     **/
    err_t compile();

    /**
     * Initializes the threading configuration of the contraction.
     *
     * @param i_num_tasks_target number of targeted tasks.
     **/
    void threading( int64_t i_num_tasks_target  );

    /**
     * Evaluates the unary operation on the given data.
     *
     * @param i_tensor_in input tensor.
     * @param io_tensor_out output tensor.
     **/
    void eval( void const * i_tensor_in,
               void       * io_tensor_out );
};

#endif
