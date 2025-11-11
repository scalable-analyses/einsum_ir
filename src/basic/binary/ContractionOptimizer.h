#ifndef EINSUM_IR_BASIC_BINARY_CONTRACTION_OPTIMIZER
#define EINSUM_IR_BASIC_BINARY_CONTRACTION_OPTIMIZER

#include <vector>
#include "../constants.h"

namespace einsum_ir {
  namespace basic {
    class ContractionOptimizer;
  }
}

class einsum_ir::basic::ContractionOptimizer {
  private:
    //! Internal enum to identify kernel iterations.
    enum{ 
      PRIM_BR = 0,
      PRIM_C  = 1,
      PRIM_M  = 2,
      PRIM_N  = 3,
      PRIM_K  = 4
    };

    //! external vector with all iterations
    std::vector< iter_property > * m_iter_space;

    //! targeted size for kernel m dimension
    int64_t m_target_m  = 0;
    //! targeted size for kernel n dimension
    int64_t m_target_n  = 0;
    //! targeted size for kernel k dimension
    int64_t m_target_k  = 0;

    //! number of bytes for scalar data types in output tensor
    int64_t m_num_bytes_scalar_out = 0;

    //! size of L2 cache in bytes
    int64_t m_l2_cache_size = 0;

    //! target size for extra packing dimensions
    int64_t m_target_extra_packing = 0;

    //! number of threads
    int64_t m_num_threads = 0;

    //! type of the main kernel
    kernel_t * m_ktype_main = nullptr;

    //! indicates if optimizer should generate sfc dimensions
    bool m_generate_sfcs = true;

    //! indicates if backend supports br gemms
    bool m_br_gemm_support = true;

    //! indicates if backend supports packing
    bool m_packing_support = true;

    //! indicates if backend supports packed gemms
    packed_gemm_t m_packed_gemm_support = packed_gemm_t::NONE;

    //! pointer to number of threads in m dimension
    int64_t * m_num_threads_sfc_m = nullptr;

    //! pointer to number of threads in n dimension
    int64_t * m_num_threads_sfc_n = nullptr;

    //! pointer to number of threads in shared dimensions
    int64_t * m_num_threads_shared = nullptr;

    //! size of the sfc in m dimension
    int64_t m_size_sfc_m = 1;

    //! size of the sfc in n dimension
    int64_t m_size_sfc_n = 1;

    /**
      * Finds all iters with a specific stride in the iteration space.
      *
      * @param o_iter_left iterator that points to the found left tensor iteration.
      * @param o_iter_right iterator that points to the found right tensor iteration.
      * @param o_iter_out iterator that points to the found output tensor iteration.
      * @param i_stride_left stride of the left tensor dimension.
      * @param i_stride_right stride of the right tensor dimension.
      * @param i_stride_out stride of the output tensor dimension.
      */
    void find_iters_with_stride( std::vector<iter_property>::iterator & o_iter_left,
                                 std::vector<iter_property>::iterator & o_iter_right,
                                 std::vector<iter_property>::iterator & o_iter_out,
                                 int64_t i_stride_left,
                                 int64_t i_stride_right,
                                 int64_t i_stride_out );
    
    /**
      * Finds an iteration with a specific dimension type. Choses the iteration with the smallest sum of strides.
      *
      * @param o_iter iterator that points to the found iteration.
      * @param i_dim_type dimension type to search for.
      **/
    void find_iter_with_dimtype( std::vector<iter_property>::iterator & o_iter,
                                 dim_t i_dim_type );
    
    /**
     * Calculates the size of all m and n dimensions in the iteration space.
     *
     * @param o_size_m size of all m dimensions. 
     * @param o_size_n size of all n dimensions.
     **/
    void get_size_all_m_n( int64_t & o_size_m,
                           int64_t & o_size_n );

    /**
      * Sets the kernel targets depending on the potential kernel size with an heuristic approach.
      *
      * @param i_potential_kernel_size potential kernel size for each dimension.
      * @param io_kernel_targets kernel targets for each dimension. 
      * @param i_iter_required indicates if at least a size 1 dimension is required.
      **/
    void set_kernel_targets_heuristic( int64_t * i_potential_kernel_size,
                                       int64_t * io_kernel_targets,
                                       bool    * i_iter_required );

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
     * Moves iters to destination vector until target size is reached. 
     * The last iteration is split to better reach target size.
     *
     * @param i_dest_iters destination vector.
     * @param i_target_size tartget size.
     * @param i_dim_type dimension type of iters to move.
     * @param i_new_exec_t execution type after moving.
     *
     * @return returns the size of all moved iters.
     **/
    int64_t move_iters_until( std::vector<iter_property> * i_dest_iters,
                              int64_t                      i_target_size,
                              dim_t                        i_dim_type,
                              exec_t                       i_new_exec_t );
    
    /**
     * Finds all divisors of a number.
     *
     * @param i_num number to find divisors for.
     * @param o_divisors output vector of divisors.
     **/                  
    void get_divisors( int64_t                i_num, 
                       std::vector<int64_t> & o_divisors );

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
     * @param i_target_m target m kernel size
     * @param i_target_n target n kernel size
     * @param i_target_k target k kernel size
     * @param i_generate_sfcs true if optimizer should generate sfc dimensions
     * @param i_br_gemm_support true if backend supports br gemms
     * @param i_packing_support true if backend supports packing
     * @param i_packed_gemm_support indicates the support level for packed gemms
     * @param i_num_bytes_scalar_out number of bytes for scalar data types in output tensor
     * @param i_l2_cache_size size of L2 cache in bytes
     * @param io_num_threads_shared number of threads used for shared parallelization.
     * @param io_num_threads_sfc_m number of threads used for sfc m parallelization.
     * @param io_num_threads_sfc_n number of threads used for sfc n parallelization.
     **/
    void init( std::vector< iter_property > * i_iter_space,
               kernel_t                     * i_ktype_main,
               int64_t                        i_target_m,
               int64_t                        i_target_n,
               int64_t                        i_target_k,
               bool                           i_generate_sfcs,
               bool                           i_br_gemm_support,
               bool                           i_packing_support,
               packed_gemm_t                  i_packed_gemm_support,                  
               int64_t                        i_num_bytes_scalar_out,
               int64_t                        i_l2_cache_size,
               int64_t                      * io_num_threads_shared,
               int64_t                      * io_num_threads_sfc_m,
               int64_t                      * io_num_threads_sfc_n );    
  
    /**
     * Optimizes the iters.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t optimize();

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
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t set_primitive_iters();

    /**
     * Reorders iters, splits iters and determines parallel iters.
     **/
    void reorder_and_parallelize_iters();
};

#endif