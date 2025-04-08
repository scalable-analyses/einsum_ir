#ifndef EINSUM_IR_BACKEND_ITERATION_SPACES_SFC
#define EINSUM_IR_BACKEND_ITERATION_SPACES_SFC

#include <vector>
#include <map>
#include "../../constants.h"


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

    //TODO 
    std::vector< dim_t >   const * m_loop_dim_type;
    std::vector< exec_t >  const * m_loop_exec_type;
    std::vector< int64_t > const * m_loop_sizes;
    std::vector< std::vector< int64_t > const * > m_loop_strides;

    int64_t m_sfc_tasks_m = 1;
    int64_t m_sfc_tasks_n = 1;
    int64_t m_sfc_tasks_k = 1;

    range_t m_parallel_loops; 

    range_t m_omp_loops;
    range_t m_sfc_loops_m;
    range_t m_sfc_loops_n;
    range_t m_sfc_loops_k;

    std::vector< range_t > m_thread_work_space;

    //! vector of movements accessed by thread_id and movement_id
    std::vector< std::vector< uint8_t > > m_dim_movements;
    
    //! vector of movement offsets accessed by ioparam_t and a movement
    std::vector< std::vector< int64_t > > m_movement_offsets;

    //! vector of initial offsets accessed by thread_id and ioparam_t
    std::vector< std::vector< int64_t > > m_initial_offsets;

    //! number of parallel loops
    int64_t m_num_parallel_loops;
    
    //! number of threads
    int64_t m_num_threads = 0;

    //! number of tasks
    int64_t m_num_tasks = 0;


    //TODO
    void convertStridesToOffsets( dim_t   i_sfc_primary,
                                  std::vector< int64_t > const & i_strides,
                                  std::vector< int64_t >       & io_offsets );
    
    //TODO
    int64_t calculateInitialOffset( int64_t i_id_omp,
                                    int64_t i_id_sfc_m,
                                    int64_t i_id_sfc_n,
                                    int64_t i_id_sfc_k,
                                    std::vector< int64_t > const & i_strides );

    //TODO
    uint8_t getMaxDimJump( range_t i_dim_loops,
                           int64_t i_id_new,
                           int64_t i_id_old );

    void SfcOracle3d( int64_t *i_m, 
                      int64_t *i_n, 
                      int64_t *i_k,
                      int64_t *i_omp, 
                      int64_t  i_idx,
                      range_t  i_thread_range );
    
    void SfcOracle2d( int64_t *i_m, 
                      int64_t *i_n, 
                      int64_t *i_omp, 
                      int64_t  i_idx );
    
    //TODO
    int gilbert_d2xy( int *x, 
                      int *y, 
                      int  idx,
                      int  w,
                      int  h );


  public:
    /**
     * Initializes the class
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
     * adds the initial offset to all datapointer
     *
     * @param i_thread_id id of thread.
     * @param i_task_id id of task
     * @param io_ptr_left pointer to pointer of left tensor
     * @param io_ptr_right pointer to pointer of right tensor
     * @param io_ptr_out pointer to pointer of output tensor
     **/
    void addMovementOffsets( int64_t          i_thread_id, 
                             int64_t          i_task_id,
                             char    const ** io_ptr_left,
                             char    const ** io_ptr_right,
                             char          ** io_ptr_out );
    
    /**
     * adds the initial offset to all datapointer
     *
     * @param i_thread_id id of thread.
     * @param io_ptr_left pointer to pointer of left tensor
     * @param io_ptr_right pointer to pointer of right tensor
     * @param io_ptr_out pointer to pointer of output tensor
     **/
    void addInitialOffsets( int64_t          i_thread_id,
                            char    const ** io_ptr_left,
                            char    const ** io_ptr_right,
                            char          ** io_ptr_out );
};

#endif