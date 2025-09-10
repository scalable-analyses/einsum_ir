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

    //! number of sfc tasks m tasks
    int64_t m_sfc_tasks_m = 1;
    //! number of sfc tasks n tasks
    int64_t m_sfc_tasks_n = 1;

    //! range of all parallel loops
    range_t m_parallel_loops; 
    //! range of omp loops
    range_t m_omp_loops;
    //! range of sfc m loops
    range_t m_sfc_loops_m;
    //! range of sfc n loops
    range_t m_sfc_loops_n;
    
    //! number of threads
    int64_t m_num_threads = 0;

    /**
     * Converts strides into offsets for sfc dimensions.
     *
     * @param io_strides vector with strides.
     **/
    void convert_strides_to_offsets( std::vector< int64_t > & io_strides );
    
    /**
     * Calculates the offset to access an element specified by the ids.
     *
     * @param i_id_omp omp id.
     * @param i_id_sfc_m sfc m id.
     * @param i_id_sfc_n sfc n id.
     * @param i_strides vector with strides.
     *
     * @return the calculated offset.
     **/
    int64_t calculate_offset( int64_t                        i_id_omp,
                              int64_t                        i_id_sfc_m,
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
     * Calculates the SFC and OMP position at the id.
     *
     * @param i_m sfc m id.
     * @param i_n sfc n id.
     * @param i_omp omp id.
     * @param i_idx task id.
     **/
    void sfc_oracle_2d( int64_t *o_m, 
                        int64_t *o_n, 
                        int64_t *o_omp, 
                        int64_t  i_idx );

  public:
    /**
     * Initializes the class.
     * restrictions:
     *    - all parallel dims must be consecutive
     *    - first omp dims can be of type m,n or c
     *    - second sfc dims of type n
     *    - third sfc dims of type m
     * example: 
     *    dim_t : ...  c1,  m1,  n1,  n2,  m2,  m3, ...
     *    exec_t: ... omp, omp, omp, sfc, sfc, sfc, ...
     *
     * @param i_loop_dim_type dimension type of the loops.
     * @param i_loop_exec_type execution type of the loops.
     * @param i_loop_sizes sizes of the loops.
     * @param i_num_threads number of threads for contraction.
    **/
    void init( std::vector< dim_t >   const * i_loop_dim_type,
               std::vector< exec_t >  const * i_loop_exec_type,
               std::vector< int64_t > const * i_loop_sizes,
               int64_t                        i_num_threads);

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
};

#endif