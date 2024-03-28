#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION_SCALAR
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION_SCALAR

#include "BinaryContraction.h"
#include "ContractionLoopsSimple.h"

namespace einsum_ir {
  namespace backend {
    class BinaryContractionScalar;
  }
}

class einsum_ir::backend::BinaryContractionScalar: public BinaryContraction {
  private:
    //! contraction loop interface
    ContractionLoopsSimple m_cont_loops;

    //! C strides of the left tensor
    std::vector< int64_t > m_strides_left_c;
    //! M strides of the left tensor
    std::vector< int64_t > m_strides_left_m;
    //! K strides of the left tensor
    std::vector< int64_t > m_strides_left_k;
    //! I strides of the left tensor
    std::vector< int64_t > m_strides_left_i;

    //! C strides of the right tensor
    std::vector< int64_t > m_strides_right_c;
    //! N strides of the right tensor
    std::vector< int64_t > m_strides_right_n;
    //! K strides of the right tensor
    std::vector< int64_t > m_strides_right_k;
    //! J strides of the right tensor
    std::vector< int64_t > m_strides_right_j;

    //! C strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_c;
    //! M strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_m;
    //! N strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_n;

    //! C strides of the output tensor
    std::vector< int64_t > m_strides_out_c;
    //! M strides of the output tensor
    std::vector< int64_t > m_strides_out_m;
    //! N strides of the output tensor
    std::vector< int64_t > m_strides_out_n;

    /**
     * Compiler-based zero kernel.
     *
     * @param_t datatype.
     * @param o_data data which is zeroed.
     **/
    template < typename T >
    static void kernel_zero( void const *,
                             void       * o_data );

    /**
     * Compiler-based ReLU kernel.
     *
     * @param_t datatype.
     * @param io_data data to which the ReLU is applied.
     **/
    template < typename T >
    static void kernel_relu( void const *,
                             void       * io_data );

    /**
     * Compiler-based copy kernel.
     *
     * @param_t datatype.
     * @param i_data_src source of the copy operation.
     * @param i_data_dst destination of the copy operation.
     **/
    template < typename T >
    static void kernel_copy( void const * i_data_src,
                             void       * io_data_dst );

    /**
     * Compiler-based multiply add kernel.
     *
     * @param_t T_LEFT data type of the left input.
     * @param_t T_RIGHT data type of the right input.
     * @param_t T_OUT data type of the output.
     **/
    template < typename T_LEFT,
               typename T_RIGHT,
               typename T_OUT >
    static void kernel_madd( void const * i_in_left,
                             void const * i_in_right,
                             void       * io_out );

    //! first-touch kernel
    void (* m_kernel_first_touch)( void const *,
                                   void       * ) = nullptr;

    //! main kernel
    void (* m_kernel_main)( void const *,
                            void const *,
                            void       * ) = nullptr;

    //! last-touch kernel
    void (* m_kernel_last_touch)( void const *,
                                  void       * ) = nullptr;

  public:
    /**
     * Compiles the binary contraction.
     * @return SUCCESS if successful, error code otherwise.
     **/
    err_t compile();

    /**
     * Not implemented.
     **/
    void threading( int64_t ){}

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_left left input tensor.
     * @param i_tensor_right right input tensor.
     * @param io_tensor_out output tensor.
     **/
    void contract( void const * i_tensor_left,
                   void const * i_tensor_right,
                   void       * io_tensor_out );

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_left left input tensor.
     * @param i_tensor_right right input tensor.
     * @param i_tensor_out_aux auxiliary data w.r.t. output tensor.
     * @param io_tensor_out output tensor.
     **/
    void contract( void const * i_tensor_left,
                   void const * i_tensor_right,
                   void const * i_tensor_out_aux,
                   void       * io_tensor_out );
};

#endif
