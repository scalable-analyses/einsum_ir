#ifndef EINSUM_IR_BASIC_UNARY_OPTIMIZER
#define EINSUM_IR_BASIC_UNARY_OPTIMIZER

#include <vector>
#include "../constants.h"

namespace einsum_ir {
  namespace basic {
    class UnaryOptimizer;
  }
}

class einsum_ir::basic::UnaryOptimizer {
  private:
   //! external vector with all iterations
   std::vector< iter_property > * m_iter_space;

   //! number of threads participating in unary opteration
   int64_t m_num_threads = 1;

  //! true if scalar execution should be generated, false otherwise
   bool m_sclar_optim = false;


  public:
   /**
     * Initializes the unary optimizer.
     *
     * @param i_iter_space vector of iters corresponding to an unoptimized unary operation.
     * @param i_num_threads number of participating threads for unary operation.
     * @param i_scalar_optim true if scalar execution should be generated, false otherwise.
     **/
    void init( std::vector< iter_property > * i_iter_space,
               int64_t                        i_num_threads, 
               bool                           i_scalar_optim );    
  
   /**
     * Optimizes the iteration space.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t optimize();
};

#endif