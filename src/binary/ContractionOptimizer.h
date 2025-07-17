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
    enum{ 
      PRIM_BR = 0,
      PRIM_C  = 1,
      PRIM_M  = 2,
      PRIM_N  = 3,
      PRIM_K  = 4
    };


    //! external vector with all iterations
    std::vector< iter_property > * m_iter_space;

    //! internal data structur with all unasigned iterations
    std::vector< std::vector< iter_property > > m_free_iters;

    //iterator pointing to the stride one dimenion in left tensor
    std::vector< iter_property >::iterator m_stride_one_left;

    //iterator pointing to the stride one dimenion in right tensor
    std::vector< iter_property >::iterator m_stride_one_right;

    //iterator pointing to the stride one dimenion in output tensor
    std::vector< iter_property >::iterator m_stride_one_out;

    //! store the combined size off all m dimension
    int64_t m_size_all_m = 0;
    //! store the combined size off all n dimension
    int64_t m_size_all_n = 0;

    //! targeted size for kernel m dimension
    int64_t m_target_m  = 0;
    //! targeted size for kernel n dimension
    int64_t m_target_n  = 0;
    //! targeted size for kernel k dimension
    int64_t m_target_k  = 0;
    //! targeted number of tasks
    int64_t m_target_parallel = 0;

    //! number of bytes for scalar data types in output tensor
    int64_t m_num_bytes_scalar_out = 0;

    //! size of L2 cache in bytes
    int64_t m_l2_cache_size = 0;

    //! number of threads
    int64_t m_num_threads = 0;

    //! type of the main kernel
    kernel_t * m_ktype_main = nullptr;

    //! indicates if backend supports br gemms
    bool m_br_gemm_support = true;

    //! indicates if backend supports packed gemms
    bool m_packed_gemm_support = true;

    //! Bounds runtine of dimension splitting but might not find the best splitting for dimensions bigger than m_max_factor.
    //! normal kernels are  smaller than 1024x1024x1024
    int64_t m_max_factor = 1024;

    //TODO
    void find_iters_with_stride( std::vector<iter_property>::iterator & o_iter_left,
                                 std::vector<iter_property>::iterator & o_iter_right,
                                 std::vector<iter_property>::iterator & o_iter_out,
                                 int64_t i_stride );
    
    void find_iter_with_dimtype( std::vector<iter_property>::iterator & o_iter,
                                 dim_t i_dim_type );
    
    void get_size_all_m_n( int64_t & o_size_m,
                           int64_t & o_size_n );

    void set_kernel_targets_heuristic( int64_t * i_potential_kernel_size,
                                       int64_t * io_kernel_targets );

    /**
     * Splits an iteration depending on a target size.
     *
     * @param i_iteration iterator that points to the iteration that is split.
     * @param i_target_size tartget size for iteration splitting
     * @param i_new_iter_pos iteration position in destination vector
     * @param i_new_exec_t execution type after splitting
     **/
    void split_iter( std::vector<iter_property>::iterator   i_iteration,
                     int64_t                                i_target_size,
                     int64_t                                i_new_iter_pos, 
                     exec_t                                 i_new_exec_t );
  
    /**
     * Adds an empty iteration to the destination vector.
     *
     * @param i_dest_iters destination vector.
     * @param i_new_iter_pos iteration position in destination vector.
     * @param i_new_dim_t dimension type of empty iteration.
     * @param i_new_exec_t execution type of empty iteration.
     **/
    void add_empty_iter( std::vector<iter_property> * i_dest_iters,
                         int64_t                      i_new_iter_pos, 
                         dim_t                        i_new_dim_t,
                         exec_t                       i_new_exec_t );

    /**
     * Moves a iteration from the source vector to the destination vector.
     *
     * @param i_iteration iterator that points to the iteration.
     * @param i_source_iters source vector.
     * @param i_dest_iters destination vector.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of the iteration.
     **/
    int64_t move_iter( std::vector<iter_property>::iterator   i_iteration,
                       std::vector<iter_property>           * i_source_iters,
                       std::vector<iter_property>           * i_dest_iters,
                       exec_t                                 i_new_exec_t );

    /**
     * Moves iters to destination vector until target size is reached. 
     * The last iteration is split to better reach target size.
     *
     * @param i_dest_iters destination vector.
     * @param i_target_size tartget size.
     * @param i_dim_type dimension type of iters to move.
     * @param i_new_exec_t execution type after moving.
     *
     **/
    void move_iters_until( std::vector<iter_property> * i_dest_iters,
                              int64_t                      i_target_size,
                              dim_t                        i_dim_type,
                              exec_t                       i_new_exec_t );
    
    /**
     * Move all remaining iters from the source vector to the destination vector.
     *
     * @param i_source_iters source vector.
     * @param i_dest_iters destination vector.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of all added iters.
     **/
    int64_t move_all_iters( std::vector<iter_property> * i_source_iters,
                            std::vector<iter_property> * i_dest_iters,
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
     * @param i_iter_space vector of iters corresponding to an unoptimized contraction.
     * @param i_ktype_main execution type of main kernel. Optimizer might change GEMMs to BR_GEMMs
     * @param i_num_threads number of participating threads in contraction.
     * @param i_target_m target m kernel size
     * @param i_target_n target n kernel size
     * @param i_target_k target k kernel size
     * @param i_br_gemm_support true if backend supports br gemms
     * @param i_packed_gemm_support true if backend supports packed gemms
     * @param i_num_bytes_scalar_out number of bytes for scalar data types in output tensor
     * @param i_l2_cache_size size of L2 cache in bytes
     **/
    void init( std::vector< iter_property > * i_iter_space,
               kernel_t                     * i_ktype_main,
               int64_t                        i_num_threads,
               int64_t                        i_target_m,
               int64_t                        i_target_n,
               int64_t                        i_target_k,
               bool                           i_br_gemm_support,
               bool                           i_packed_gemm_support,
               int64_t                        i_num_bytes_scalar_out,
               int64_t                        i_l2_cache_size );    
  
    /**
     * Optimizes the iters.
     **/
    void optimize();

    /**
     * Sorts all external iters depending on stride, dimension type and execution type.
     **/
    void sort_and_fuse_iters();

    /**
     * Removes all size 1 iters that are not of primitive type
     **/
    void remove_empty_iters();

    /**
     * Moves all unoptimized external iters to internal data structure.
     **/
    void move_iters_to_internal();

    /**
     * Finds and adds a kernel to the optimized iters.
     **/
    err_t set_primitive_iters();

    /**
     * Reorders iters, splits iters and determines parallel iters.
     **/
    void reorder_and_parallelize_iters();
};

#endif