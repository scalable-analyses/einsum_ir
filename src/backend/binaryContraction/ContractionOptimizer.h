#ifndef EINSUM_IR_BACKEND_CONTRACTION_OPTIMIZER
#define EINSUM_IR_BACKEND_CONTRACTION_OPTIMIZER

#include <vector>
#include <map>
#include "../../constants.h"
#include "constants_local.h"
#include "IterationSpacesSfc.h"


namespace einsum_ir {
  namespace backend {
    class ContractionOptimizer;
  }
}

class einsum_ir::backend::ContractionOptimizer {
  private:
    std::vector< loop_property > * m_loops;

    //! kernel targets
    int64_t m_target_m  = 16;
    int64_t m_target_n  = 64;
    int64_t m_target_k  = 256;

    int64_t m_target_parallel = 16384;

    int64_t m_num_tasks = 72;

    //! type of the main kernel
    kernel_t * m_ktype_main;

    int64_t splitLoop( std::vector<loop_property>::iterator i_loop,
                       std::vector<loop_property> * i_source_loops,
                       std::vector<loop_property> * i_dest_loops,
                       int64_t i_target_size,
                       int64_t i_new_loop_pos, 
                       exec_t  i_new_exec_t );
                    
    int64_t addLoop( std::vector<loop_property>::iterator i_loop,
                     std::vector<loop_property> * i_source_loops,
                     std::vector<loop_property> * i_dest_loops,
                     int64_t i_new_loop_pos, 
                     exec_t  i_new_exec_t );

    int64_t add_loops_until( std::vector<loop_property> * i_source_loops,
                             std::vector<loop_property> * i_dest_loops,
                             int64_t i_target_size,
                             exec_t  i_new_exec_t );
    
    int64_t add_all_loops( std::vector<loop_property> * i_source_loops,
                           std::vector<loop_property> * i_dest_loops,
                           exec_t  i_new_exec_t );  
            

    int64_t findSplit( int64_t i_dim_size,
                       int64_t i_target_size );

  public:
    void init( std::vector< loop_property > * i_loops,
               kernel_t                     * i_ktype_main );    
  
    void optimize();

    void sortLoops();

    void fuseLoops();

    void reorderLoops();

};

#endif