#ifndef EINSUM_IR_BASIC_BINARY_ITERATION_SPACE
#define EINSUM_IR_BASIC_BINARY_ITERATION_SPACE

#include <vector>
#include "../constants.h"

namespace einsum_ir {
  namespace basic {
    class IterationSpace;
  }
}

class einsum_ir::basic::IterationSpace {
  private:
    struct range_t{
      int64_t begin = 0;
      int64_t end = 0;
    };

    //! vector with the loop dimension types
    std::vector< dim_t >   const * m_dim_types;
    //! vector with the loop execution types
    std::vector< exec_t >  const * m_exec_types;
    //! vector with the loop sizes
    std::vector< int64_t > const * m_sizes;

    //! number of shared tasks
    int64_t m_shared_tasks = 1;
    //! number of sfc m tasks
    int64_t m_sfc_tasks_m = 1;
    //! number of sfc n tasks
    int64_t m_sfc_tasks_n = 1;
    //! number of sfc k tasks
    int64_t m_sfc_tasks_k = 1;

    //! range of all shared loops
    range_t m_shared_loops; 
    //! range of sfc m loops
    range_t m_sfc_loops_m;
    //! range of sfc n loops
    range_t m_sfc_loops_n;
    //! range of sfc k loops
    range_t m_sfc_loops_k;
    
    //! number of threads in shared dimensions
    int64_t m_num_threads_shared = 0;
    //! number of threads in sfc m dimension
    int64_t m_num_threads_m = 0;
    //! number of threads in sfc n dimension
    int64_t m_num_threads_n = 0;

    //! number of tasks in shared dimensions
    int64_t m_tasks_per_thread_shared;
    //! number of tasks in m dimension 
    int64_t m_tasks_per_thread_m;
    //! number of tasks in n dimension 
    int64_t m_tasks_per_thread_n;

    /**
     * Converts strides into offsets for sfc dimensions.
     *
     * @param io_strides vector with strides.
     **/
    void convert_strides_to_offsets( std::vector< int64_t > & io_strides );
    
    /**
     * Calculates the offset to access an element specified by the ids.
     *
     * @param i_id_sfc_m sfc m id.
     * @param i_id_sfc_n sfc n id.
     * @param i_strides vector with strides.
     *
     * @return the calculated offset.
     **/
    int64_t calculate_offset( int64_t                        i_id_sfc_m,
                              int64_t                        i_id_sfc_n,
                              std::vector< int64_t > const & i_strides );

    /**
     * Calculates the movement direction for a tensor contraction from old and new id.
     *
     * @param i_dim_loops loops that correspond to id change e.g. range of sfc_m loops.
     * @param i_id_new new id.
     * @param i_id_old old id.
     *
     * @return the movement.
     **/
    sfc_t get_max_dim_jump( range_t i_dim_loops,
                            int64_t i_id_new,
                            int64_t i_id_old );
    
    /**
     * Calculates the position in a 2d SFC from the given id.
     *
     * @param i_idx task id.
     * @param i_sfc_size_m size of sfc in m direction.
     * @param i_sfc_size_n size of sfc in n direction.
     * @param o_m sfc m id.
     * @param o_n sfc n id.
     **/
    void sfc_oracle_2d( int64_t  i_idx,
                        int64_t  i_sfc_size_m,
                        int64_t  i_sfc_size_n,
                        int64_t *o_m, 
                        int64_t *o_n);
    
    /**
     * Calculates the position in a semi 3d SFC by combining two 2d SFCs.
     *
     * @param i_idx task id.
     * @param i_sfc_size_m size of sfc in m direction.
     * @param i_sfc_size_n size of sfc in n direction.
     * @param i_sfc_size_k size of sfc in k direction.
     * @param o_m sfc m id.
     * @param o_n sfc n id.
     * @param o_k sfc k id.
     **/
    void sfc_oracle_3d( int64_t  i_idx,
                        int64_t  i_sfc_size_m,
                        int64_t  i_sfc_size_n,
                        int64_t  i_sfc_size_k,
                        int64_t *o_m, 
                        int64_t *o_n,
                        int64_t *o_k );

  public:
    /**
     * Initializes the class.
     * restrictions:
     *    - all dimensions marked as SFC must be consecutive
     *    - first  sfc dims of type n
     *    - second sfc dims of type m
     *    - third  sfc dims of type k
     * example: 
     *    dim_t : ...   n1, n2, m1, m2, m3, k1, k2 ...
     *
     * @param i_loop_dim_type dimension type of the loops.
     * @param i_loop_exec_type execution type of the loops.
     * @param i_loop_sizes sizes of the loops.
     * @param i_num_threads_m number of threads in sfc m dimension.
     * @param i_num_threads_n number of threads in sfc n dimension.
     * @param i_num_threads_shared number of threads in shared dimensions.
    **/
    void init( std::vector< dim_t >   const * i_loop_dim_type,
               std::vector< exec_t >  const * i_loop_exec_type,
               std::vector< int64_t > const * i_loop_sizes,
               int64_t                        i_num_threads_m,
               int64_t                        i_num_threads_n,
               int64_t                        i_num_threads_shared );

    /**
     * Creates ThreadInfo objects for each thread and changes sfc strides.
     *
     * @param io_loop_strides_left strides in the left input tensor.
     * @param io_loop_strides_right strides in the right input tensor.
     * @param io_loop_strides_out_aux strides in the auxiliary output tensor.
     * @param io_loop_strides_out strides in the output tensor.
     * @param io_thread_infos vector to store thread information.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t setup( std::vector< int64_t >   & io_strides_left,
                 std::vector< int64_t >   & io_strides_right,
                 std::vector< int64_t >   & io_strides_out_aux,
                 std::vector< int64_t >   & io_strides_out,
                 std::vector<thread_info> & io_thread_infos );

    /**
     * Simple function to determine if caching of values could be advantageous.
     *
     * @return 1 (don't use extra memory) if no advantage, higher number of cached values otherwise.
     **/
    int64_t get_caching_size();

};

#endif