#ifndef EINSUM_IR_BINARY_UNARY_OPTIMIZER
#define EINSUM_IR_BINARY_UNARY_OPTIMIZER

#include <vector>
#include "../constants.h"

namespace einsum_ir {
  namespace binary {
    class UnaryOptimizer;
  }
}

class einsum_ir::binary::UnaryOptimizer {
  private:
   //! external vector with all iterations
   std::vector< iter_property > * m_iter_space;

   //! number of threads participating in unary opteration
   int64_t m_num_threads = 1;


  public:
   /**
     * Initializes the unary optimizer.
     *
     * @param i_iter_space vector of iters corresponding to an unoptimized contraction.
     * @param i_num_threads number of participating threads in contraction.
     **/
    void init( std::vector< iter_property > * i_iter_space,
               int64_t                        i_num_threads
             );    
  
   /**
     * Optimizes the iteration space.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t optimize();
};

#endif