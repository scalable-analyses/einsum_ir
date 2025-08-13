#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION_SCALAR
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION_SCALAR

#include "BinaryContraction.h"
#include "../basic/binary/ContractionBackendScalar.h"

namespace einsum_ir {
  namespace backend {
    class BinaryContractionScalar;
  }
}

class einsum_ir::backend::BinaryContractionScalar: public BinaryContraction {
  private:
    //! target for the primitive m dimension
    int64_t m_target_prim_m = 1;

    //! target for the primitive n dimension
    int64_t m_target_prim_n = 1;

    //! target for the primitive k dimension
    int64_t m_target_prim_k = 1;
  
     //! contraction backend
    einsum_ir::basic::ContractionBackendScalar m_backend;

    /**
     * Helper function for map find with default value
     *
     * @param i_map map.
     * @param i_key key.
     * @param i_default default value.
     *
     * @param return value or default value.
     **/
    template <typename T>
    T map_find_default( std::map< int64_t, T > const * i_map,
                        int64_t                        i_key,
                        T                              i_default ){
      if( auto search = i_map->find(i_key); search != i_map->end() ) {
        return search->second;
      }
      else {
        return i_default;
      }
    }

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
