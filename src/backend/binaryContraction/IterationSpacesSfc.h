#ifndef EINSUM_IR_BACKEND_ITERATION_SPACES_SFC
#define EINSUM_IR_BACKEND_ITERATION_SPACES_SFC

#include <vector>
#include <map>
#include "constants_local.h"


namespace einsum_ir {
  namespace backend {
    class IterationSpacesSfc;
  }
}

class einsum_ir::backend::IterationSpacesSfc {
  private:
    struct range_t{
      int64_t begin = 0;
      int64_t end = 0;
    };

    //! vector with the loop dimension types
    std::vector< dim_t >   const * m_loop_dim_type;
    //! vector with the loop execution types
    std::vector< exec_t >  const * m_loop_exec_type;
    //! vector with the loop sizes
    std::vector< int64_t > const * m_loop_sizes;
    //! vector with vectors for all strides
    std::vector< std::vector< int64_t > const * > m_loop_strides;

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

    //! matrix of movements accessed by thread_id and task_id
    std::vector< std::vector< uint8_t > > m_dim_movements;
    
    //! matrix of movement offsets accessed by io_id and movement_id
    std::vector< std::vector< int64_t > > m_movement_offsets;

    //! matrix of initial offsets accessed by thread_id and io_id
    std::vector< std::vector< int64_t > > m_initial_offsets;
    
    //! number of threads
    int64_t m_num_threads = 0;

    /**
     * Converts strides into offsets from previous dimension.
     * i.e. creates a data structure that encodes jumps corresponding to different movements through tensor
     *
     * @param i_strides vector with strides.
     * @param io_offsets vector with calculated offsets.
     **/
    void convertStridesToOffsets( std::vector< int64_t > const & i_strides,
                                  std::vector< int64_t >       & io_offsets );
    
    /**
     * Calculates the offset to access an element specified by the ids.
     *
     * @param i_id_omp omp id.
     * @param i_id_sfc_m sfc m id.
     * @param i_id_sfc_n sfc n id.
     *
     * @return the calculated offset.
     **/
    int64_t calculateOffset( int64_t i_id_omp,
                             int64_t i_id_sfc_m,
                             int64_t i_id_sfc_n,
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
    uint8_t getMaxDimJump( range_t i_dim_loops,
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
    void SfcOracle2d( int64_t *i_m, 
                      int64_t *i_n, 
                      int64_t *i_omp, 
                      int64_t  i_idx );
    
    /**
     * calculates gilbert curve
     *
     * @param x calculated x id.
     * @param y calculated y id.
     * @param idx sfc id.
     * @param w sfc width.
     * @param h sfc height
     **/
    void gilbert_d2xy( int *x, 
                       int *y, 
                       int  idx,
                       int  w,
                       int  h );


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
     * @param i_loop_strides_left strides in the left input tensor.
     * @param i_loop_strides_right strides in the right input tensor.
     * @param i_loop_strides_out_aux strides in the auxiliary output tensor.
     * @param i_loop_strides_out strides in the output tensor.
     * @param i_num_threads number of threads for contraction.
    **/
    void init( std::vector< dim_t >   const * i_loop_dim_type,
               std::vector< exec_t >  const * i_loop_exec_type,
               std::vector< int64_t > const * i_loop_sizes,
               std::vector< int64_t > const * i_loop_strides_left,
               std::vector< int64_t > const * i_loop_strides_right,
               std::vector< int64_t > const * i_loop_strides_out_aux,
               std::vector< int64_t > const * i_loop_strides_out,
               int64_t                        i_num_threads);


    /**
     * Compiles the contraction loop interface.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile();

    /**
     * Gets the number of tasks for a specified thread.
     *
     * @param i_thread_id id of thread.
     *
     * @return number of tasks.
     **/
    int64_t getNumTasks( int64_t i_thread_id );

    /**
     * Adds the Movement at the task id to all datapointer.
     *
     * @param i_thread_id id of thread.
     * @param i_task_id id of task.
     * @param io_ptr_left pointer to pointer of left tensor.
     * @param io_ptr_right pointer to pointer of right tensor.
     * @param io_ptr_out pointer to pointer of output tensor.
     **/
    void addMovementOffsets( int64_t          i_thread_id, 
                             int64_t          i_task_id,
                             char    const ** io_ptr_left,
                             char    const ** io_ptr_right,
                             char          ** io_ptr_out );
    
    /**
     * Gets the initial offsets for all datapointer.
     *
     * @param i_thread_id id of thread.
     * @param io_ptr_left pointer to pointer of left tensor.
     * @param io_ptr_right pointer to pointer of right tensor.
     * @param io_ptr_out pointer to pointer of output tensor.
     **/
    void getInitialOffsets( int64_t   i_thread_id,
                            int64_t & io_off_left,
                            int64_t & io_off_right,
                            int64_t & io_off_out );
};

#endif