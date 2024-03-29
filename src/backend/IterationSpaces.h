#ifndef EINSUM_IR_BACKEND_ITERATION_SPACES
#define EINSUM_IR_BACKEND_ITERATION_SPACES

#include <vector>
#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class IterationSpaces;
  }
}

class einsum_ir::backend::IterationSpaces {
  private:
    //! number of loops
    int64_t m_num_loops = -1;

    //! number of outer loops which may be executed in parallel
    int64_t m_num_parallel = -1;

    //! targeted number of tasks
    int64_t m_num_tasks_target = -1;

    //! number of tasks
    int64_t m_num_tasks = -1;

    //! number of collapsed loops
    int64_t m_num_collapsed = -1;

    /**
     * Parameters of a single loop. 
     **/
    typedef struct {
      //! first elements
      std::vector< int64_t > firsts;
      //! sizes of the loops
      std::vector< int64_t > sizes;
    } IterSpace;

    //! global iteration space
    IterSpace m_global_space;

    //! thread-local iteration spaces
    std::vector< IterSpace > m_thread_local_spaces;

    //! true if the iteration spaces interface was compiled
    bool m_compiled = false;

  public:
    /**
     * Initializes the iteration spaces interface. 
     *
     * @param i_num_loops number of nested loops.
     * @param i_num_parallel number of outer loops which may be executed in parallel.
     * @param i_firsts first iteration performed in the respective nested loop. pass nullptr if 0 for all loops.
     * @param i_sizes sizes of the loops.
     * @param i_num_tasks number of tasks into which the nested loops are split.
     **/
    void init( int64_t         i_num_loops,
               int64_t         i_num_parallel,
               int64_t const * i_firsts,
               int64_t const * i_sizes,
               int64_t         i_num_tasks );

    /**
     * Compiles the iteration spaces interface.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile();

    /**
     * Gets the number of collapsed loops.
     * The first (num_collapsed - 1) loops have size 1.
     *
     * @return number of collapsed loops.
     **/
    int64_t num_collapsed();

    /**
     * Gets the number of tasks.
     *
     * @return number of tasks
     **/
    int64_t num_tasks();

    /**
     * Gets the first elements for the nested loops for a given tasks.
     *
     * @param i_task_id id of the task.
     * @return first elements. 
     **/
    int64_t const * firsts( int64_t i_task_id );

    /**
     * Gets the sizes of the nested loops for a given tasks.
     *
     * @param i_task_id id of the task.
     * @return loop sizes. 
     **/
    int64_t const * sizes(  int64_t i_task_id );
};

#endif