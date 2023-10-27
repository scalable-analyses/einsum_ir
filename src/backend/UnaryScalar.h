#ifndef EINSUM_IR_BACKEND_UNARY_SCALAR
#define EINSUM_IR_BACKEND_UNARY_SCALAR

#include "Unary.h"
#include "UnaryLoopsSimple.h"

namespace einsum_ir {
  namespace backend {
    class UnaryScalar;
  }
}

class einsum_ir::backend::UnaryScalar: public Unary {
  private:
    //! unary loop interface
    UnaryLoopsSimple m_unary_loops;

    /**
     * Compiler-based copy kernel.
     *
     * @param_t T datatype.
     * @param i_data_src source of the copy operation.
     * @param io_data_dst destination of the copy operation.
     **/
    template < typename T >
    static void kernel_copy( void const * i_data_src,
                             void       * io_data_dst );

    //! main kernel
    void (* m_kernel_main)( void const *,
                            void       * ) = nullptr;

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
