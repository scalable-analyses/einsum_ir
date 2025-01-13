#ifndef EINSUM_IR_BACKEND_ITERATION_SPACES_SFC
#define EINSUM_IR_BACKEND_ITERATION_SPACES_SFC

#include <vector>
#include <map>
#include "../constants.h"


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

    typedef enum {
      LEFT = 0, // left input
      RIGHT = 1, // right input
      OUT = 2, // output
      MAX_NUM_INPUTS = 3 //
    } input_t;

    //TODO 
    std::vector< dim_t >   const * m_loop_dim_type;
    std::vector< exec_t >  const * m_loop_exec_type;
    std::vector< int64_t > const * m_loop_sizes;
    std::vector< std::vector< int64_t > const * > m_loop_strides;

    //std::vector< int64_t > const * m_loop_strides_left;
    //std::vector< int64_t > const * m_loop_strides_right;
    //std::vector< int64_t > const * m_loop_strides_out;


    //! vector of movements accessed by thread_id and movement_id
    std::vector< std::vector< uint8_t > > m_dim_movements;
    
    //! vector of movement offsets accessed by input_t and a movement
    std::vector< std::vector< int64_t > > m_movement_offsets;

    //! vector of initial offsets accessed by thread_id and input_t
    std::vector< std::vector< int64_t > > m_initial_offsets;

    //! number of parallel loops
    int64_t m_num_parallel_loops;
    
    //! number of threads
    int64_t m_num_threads = 0;

    //! number of tasks
    int64_t m_num_tasks = 0;


    //TODO
    void convertStridesToOffsets( range_t i_omp_loops,
                                  range_t i_sfc_loops_m,
                                  range_t i_sfc_loops_n,
                                  bool    i_sfc_m_large,
                                  std::vector< int64_t > const & i_strides,
                                  std::vector< int64_t >       & io_offsets );
    
    //TODO
    int64_t calculateInitialOffsets( range_t i_omp_loops,
                                     range_t i_sfc_loops_m,
                                     range_t i_sfc_loops_n,
                                     int64_t i_id_omp,
                                     int64_t i_id_sfc_m,
                                     int64_t i_id_sfc_n,
                                     std::vector< int64_t > const & i_strides );

    //TODO
    uint8_t getMaxDimJump( range_t i_dim_loops,
                           int64_t i_id_new,
                           int64_t i_id_old,
                           int64_t i_offset );

    //TODO
    int gilbert_d3xy(int *x, 
                      int *y, 
                      int *z, 
                      int idx, 
                      int w, 
                      int h );


  public:
    /**
     * TODO
     * assigns parallel dimensions to three types omp, sfc_n, sfc_m
     * restrictions:
     *   all parallel dims must be consecutive
     *  first omp dims can be of type m,n or c
     *  second sfc dims of type n
     *  third sfc dims of type m
     * example: 
     *  dim_t : ...  c1,  m1,  n1,  n2,  m2,  m3, ...
     *  exec_t: ... omp, omp, omp, sfc, sfc, sfc, ...
    **/
    void init( std::vector< dim_t >   const * i_loop_dim_type,
               std::vector< exec_t >  const * i_loop_exec_type,
               std::vector< int64_t > const * i_loop_sizes,
               std::vector< int64_t > const * i_loop_strides_left,
               std::vector< int64_t > const * i_loop_strides_right,
               std::vector< int64_t > const * i_loop_strides_out,
               int64_t                        i_num_threads);


    /**
     * Compiles the contraction loop interface.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile();

    /**
     * Gets the number of tasks.
     *
     * @return number of tasks
     **/
    int64_t num_tasks();

    /**
     * TODO
     **/
    int64_t getNumTasks( int64_t i_thread_id );

    /**
     * TODO
     **/
    void addMovementOffsets( int64_t    i_thread_id, 
                             int64_t    i_task_id,
                             char    ** io_offset_left,
                             char    ** io_offset_right,
                             char    ** io_offset_out );
    
    /**
     * TODO
     **/
    void addInitialOffsets( int64_t    i_thread_id,
                            char    ** io_offset_left,
                            char    ** io_offset_right,
                            char    ** io_offset_out );
};

#endif