#ifndef EINSUM_IR_BINARY_CONTRACTION_OPTIMIZER
#define EINSUM_IR_BINARY_CONTRACTION_OPTIMIZER

#include <vector>
#include "constants.h"

namespace einsum_ir {
  namespace binary {
    class ContractionOptimizer;
  }
}

class einsum_ir::binary::ContractionOptimizer {
  private:
    //! external vector with all iterations
    std::vector< iter_property > * m_iter_space;

    //! internal data structur with all unasigned iterations
    std::vector< std::vector< iter_property > > m_free_iters;

    //! store the combined size off all m dimension
    int64_t m_size_all_m;
    //! store the combined size off all n dimension
    int64_t m_size_all_n;

    //! targeted size for kernel m dimension
    int64_t m_target_m  = 1;
    //! targeted size for kernel n dimension
    int64_t m_target_n  = 1;
    //! targeted size for kernel k dimension
    int64_t m_target_k  = 1;
    //! targeted number of tasks
    int64_t m_target_parallel = 1;

    //! number of threads
    int64_t m_num_threads = 1;

    //! type of the main kernel
    kernel_t * m_ktype_main = nullptr;

    //! indicates if backend supports br gemms
    bool m_br_gemm_support = true;

    //! indicates if backend supports packed gemms
    bool m_packed_gemm_support = true;

    //! Bounds runtine of dimension splitting but might not find the best splitting for dimensions bigger than m_max_factor.
    //! normal kernels are  smaller than 1024x1024x1024
    int64_t m_max_factor = 1024;

    /**
     * Splits a loop from the source vector and adds it to the destination vector.
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
    int64_t split_loop( std::vector<iter_property>::iterator   i_loop,
                        std::vector<iter_property>           * i_source_loops,
                        std::vector<iter_property>           * i_dest_loops,
                        int64_t                                i_target_size,
                        int64_t                                i_new_loop_pos, 
                        exec_t                                 i_new_exec_t );
  
    /**
     * Adds an empty loop to the destination vector.
     *
     * @param i_dest_loops destination vector.
     * @param i_new_loop_pos loop position in destination vector.
     * @param i_new_dim_t dimension type of empty loop.
     * @param i_new_exec_t execution type of empty loop.
     **/
    void add_empty_loop( std::vector<iter_property> * i_dest_loops,
                         int64_t                      i_new_loop_pos, 
                         dim_t                        i_new_dim_t,
                         exec_t                       i_new_exec_t );

    /**
     * Moves a loop from the source vector to the destination vector.
     *
     * @param i_loop iterator that points to the loop.
     * @param i_source_loops source vector.
     * @param i_dest_loops destination vector.
     * @param i_new_loop_pos loop position in destination vector.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of the loop.
     **/
    int64_t move_loop( std::vector<iter_property>::iterator   i_loop,
                       std::vector<iter_property>           * i_source_loops,
                       std::vector<iter_property>           * i_dest_loops,
                       int64_t                                i_new_loop_pos, 
                       exec_t                                 i_new_exec_t );

    /**
     * Moves loops to destination vector until target size is reached. 
     * The last loop is split to better reach target size.
     *
     * @param i_source_loops source vector.
     * @param i_dest_loops destination vector.
     * @param i_target_size tartget size.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of all added loops.
     **/
    int64_t move_loops_until( std::vector<iter_property> * i_source_loops,
                              std::vector<iter_property> * i_dest_loops,
                              int64_t                      i_target_size,
                              exec_t                       i_new_exec_t );
    
    /**
     * Move all remaining loops from the source vector to the destination vector.
     *
     * @param i_source_loops source vector.
     * @param i_dest_loops destination vector.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of all added loops.
     **/
    int64_t move_all_loops( std::vector<iter_property> * i_source_loops,
                           std::vector<iter_property> * i_dest_loops,
                           exec_t                       i_new_exec_t );  
            

    /**
     * Determines a good integer splitt for a dimension size to be close to the target size.
     *
     * @param i_dim_size dimension size before split.
     * @param i_target_size target size.
     *
     * @return returns a integer split.
     **/                  
    int64_t find_split( int64_t i_dim_size,
                        int64_t i_target_size );

  public:
   /**
     * Initializes the contraction optimizer.
     *
     * @param i_iter_space vector of loops corresponding to an unoptimized contraction.
     * @param i_ktype_main execution type of main kernel. Optimizer might change GEMMs to BR_GEMMs
     * @param i_num_threads number of participating threads in contraction.
     * @param i_target_m target m kernel size
     * @param i_target_n target n kernel size
     * @param i_target_k target k kernel size
     * @param i_br_gemm_support true if backend supports br gemms
     * @param i_packed_gemm_support true if backend supports packed gemms
     **/
    void init( std::vector< iter_property > * i_iter_space,
               kernel_t                     * i_ktype_main,
               int64_t                        i_num_threads,
               int64_t                        i_target_m,
               int64_t                        i_target_n,
               int64_t                        i_target_k,
               bool                           i_br_gemm_support,
               bool                           i_packed_gemm_support  );    
  
    /**
     * Optimizes the loops.
     **/
    void optimize();

    /**
     * Sorts all external loops depending on stride, dimension type and execution type.
     **/
    void sort_loops();

    /**
     * Removes all size 1 Loops that are not of primitive type
     **/
    void remove_empty_loops();

    /**
     * Fuses all external loops with the same dimension type, execution type and contiguous storage.
     **/
    void fuse_loops();

    /**
     * Moves all unoptimized external loops to internal data structure.
     **/
    void move_loops_to_internal();

    /**
     * Finds and adds a kernel to the optimized loops.
     **/
    void add_kernel();

    /**
     * Reorders loops, splits loops and determines parallel loops.
     **/
    void reorder_loops();
};

#endif