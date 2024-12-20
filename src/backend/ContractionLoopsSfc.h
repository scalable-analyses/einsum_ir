#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS_SFC
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS_SFC

#include <cstdint>
#include <vector>
#include <map>
#include "../constants.h"
#include "IterationSpaces.h"


namespace einsum_ir {
  namespace backend {
    class ContractionLoopsSFC;
  }
}

class einsum_ir::backend::ContractionLoopsSFC {
  private:
    std::vector< dim_t >   const m_loop_dim_type;
    std::vector< exec_t >  const m_loop_exec_type;
    std::vector< int64_t > const m_loop_sizes;
    std::vector< int64_t > const m_loop_strides_left;
    std::vector< int64_t > const m_loop_strides_right;
    std::vector< int64_t > const m_loop_strides_out_aux;
    std::vector< int64_t > const m_loop_strides_out;

  public:
    void init( std::vector< dim_t >   const & i_loop_dim_type,
               std::vector< exec_t >  const & i_loop_exec_type,
               std::vector< int64_t > const & i_loop_sizes,
               std::vector< int64_t > const & i_loop_strides_left,
               std::vector< int64_t > const & i_loop_strides_right,
               std::vector< int64_t > const & i_loop_strides_out_aux,
               std::vector< int64_t > const & i_loop_strides_out );


    /**
     * Compiles the contraction loop interface.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile();
};

#endif