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
    //! external vector with all loops 
    std::vector< loop_property > * m_loops;

    //! internal data structur with all loops
    std::vector< std::vector< loop_property > > m_free_loops;

    //! store the combined size off all m dimension
    int64_t m_size_all_m;
    //! store the combined size off all n dimension
    int64_t m_size_all_n;

    //! targeted size for kernel m dimension
    int64_t m_target_m  = 16;
    //! targeted size for kernel n dimension
    int64_t m_target_n  = 64;
    //! targeted size for kernel k dimension
    int64_t m_target_k  = 256;
    //! targeted number of tasks
    int64_t m_target_parallel = 16384;

    //! number of threads
    int64_t m_num_threads;

    //! type of the main kernel
    kernel_main_t * m_ktype_main;

    /**
     * splits a loop from the source vector and adds it to the destination vector.
     *
     * @param i_loop iterator that points to the loop that is split.
     * @param i_source_loops source vector.
     * @param i_dest_loops destination vector.
     * @param i_target_size tartget size for loop splitting
     * @param i_new_loop_pos loop position in destination vector
     * @param i_new_exec_t execution type after splitting
     *
     * @return returns the size of the splitted loop.
     **/
    int64_t splitLoop( std::vector<loop_property>::iterator i_loop,
                       std::vector<loop_property> * i_source_loops,
                       std::vector<loop_property> * i_dest_loops,
                       int64_t i_target_size,
                       int64_t i_new_loop_pos, 
                       exec_t  i_new_exec_t );
  
    /**
     * push an empty loop to the back of a destination vector.
     *
     * @param i_dest_loops destination vector.
     * @param i_new_dim_t dimension type of empty loop
     **/
    void push_empty_loop( std::vector<loop_property> * i_dest_loops,
                          dim_t                        i_new_dim_t );

    /**
     * adds a loop from the source vector to the destination vector.
     *
     * @param i_loop iterator that points to the loop.
     * @param i_source_loops source vector.
     * @param i_dest_loops destination vector.
     * @param i_new_loop_pos loop position in destination vector.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of the loop.
     **/
    int64_t addLoop( std::vector<loop_property>::iterator i_loop,
                     std::vector<loop_property> * i_source_loops,
                     std::vector<loop_property> * i_dest_loops,
                     int64_t i_new_loop_pos, 
                     exec_t  i_new_exec_t );

    /**
     * add loop to destination vector until target size is reached. 
     * the last loop is split to better reach target size.
     *
     * @param i_source_loops source vector.
     * @param i_dest_loops destination vector.
     * @param i_target_size tartget size.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of all added loops.
     **/
    int64_t add_loops_until( std::vector<loop_property> * i_source_loops,
                             std::vector<loop_property> * i_dest_loops,
                             int64_t i_target_size,
                             exec_t  i_new_exec_t );
    
    /**
     * adds all remaining loops from the source vector to the destination vector.
     *
     * @param i_source_loops source vector.
     * @param i_dest_loops destination vector.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of all added loops.
     **/
    int64_t add_all_loops( std::vector<loop_property> * i_source_loops,
                           std::vector<loop_property> * i_dest_loops,
                           exec_t  i_new_exec_t );  
            

    /**
     * determines a good integer splitt for a dimension size to be close to the target size.
     *
     * @param i_dim_size dimension size before split.
     * @param i_target_size target size.
     *
     * @return returns a integer split.
     **/                  
    int64_t findSplit( int64_t i_dim_size,
                       int64_t i_target_size );

  public:
   /**
     * Initializes the contraction optimizer.
     *
     * @param i_loops vector of loops corresponding to an unoptimized contraction.
     * @param i_ktype_main execution type of main kernel. Optimizer might change GEMMs to BR_GEMMs
     * @param i_num_threads number of participating threads in contraction.
     **/
    void init( std::vector< loop_property > * i_loops,
               kernel_main_t                * i_ktype_main,
               int64_t                        i_num_threads );    
  
    /**
     * optimizes the loops.
     **/
    void optimize();

    /**
     * sorts all external loops depending on stride, dimension type and execution type.
     **/
    void sortLoops();

    /**
     * fuses all external loops with the same dimension type, execution type and contiguous storage.
     **/
    void fuseLoops();

    /**
     * moves all unoptimized external loops to internal data structure.
     **/
    void moveLoopsToInternal();


    /**
     * finds and adds a kernel to the optimized loops.
     **/
    void addKernel();

    /**
     * reorders loops, splits loops and determines parallel loops.
     **/
    void reorderLoops();
};

#endif