#ifndef EINSUM_IR_BACKEND_CONTRACTION_LOOPS
#define EINSUM_IR_BACKEND_CONTRACTION_LOOPS

#include <cstdint>
#include <vector>
#include <map>
#include "../constants.h"
#include "IterationSpaces.h"
#include "ContractionPackingTpp.h"

namespace einsum_ir {
  namespace backend {
    class ContractionLoops;
  }
}

class einsum_ir::backend::ContractionLoops {
  private:
    //! number of C dimensions
    int64_t m_num_dims_c = 0;
    //! number of M dimensions
    int64_t m_num_dims_m = 0;
    //! number of N dimensions
    int64_t m_num_dims_n = 0;
    //! number of K dimensions
    int64_t m_num_dims_k = 0;

    //! Ids of C dimension
    int64_t const * m_dim_ids_c = nullptr;
    //! Ids of M dimension
    int64_t const * m_dim_ids_m = nullptr;
    //! Ids of N dimension
    int64_t const * m_dim_ids_n = nullptr;
    //! Ids of K dimension
    int64_t const * m_dim_ids_k = nullptr;

    //! vector of loop execution order
    std::vector<int64_t> * m_loop_ids = nullptr;

    //! mapping from dimension id to type
    std::map< int64_t, dim_t > const * m_dim_type = nullptr;

    //! sizes of dimensions
    std::map< int64_t, int64_t > const * m_sizes = nullptr;

    //! strides of the left input tensor
    std::map< int64_t, int64_t > const * m_strides_left = nullptr;

    //! strides of the right input tensor
    std::map< int64_t, int64_t > const * m_strides_right = nullptr;

    //! strides of the auxiliary output tensor
    std::map< int64_t, int64_t > const * m_strides_out_aux = nullptr;

    //! strides of the output tensor
    std::map< int64_t, int64_t > const * m_strides_out = nullptr;

    //! number of bytes for a scalar of the left input tensor
    int64_t m_num_bytes_scalar_left = 0;
    //! number of bytes for a scalar of the right input tensor
    int64_t m_num_bytes_scalar_right = 0;
    //! number of bytes for a scalar of the output tensor
    int64_t m_num_bytes_scalar_out = 0;

    //! number of targeted tasks
    int64_t m_num_tasks_targeted = 0;

    //! number of tasks
    int64_t m_num_tasks = 0;

    //! id of first parallel loop
    int64_t m_id_first_parallel = 0;

    //! iteration spaces
    IterationSpaces m_iter_spaces;

    //! id of packing loop for left input
    int64_t m_id_loop_packing_left = 0;

    //! id of packing loop for right input
    int64_t m_id_loop_packing_right = 0;

    //! id of first/last touch loop
    int64_t m_id_loop_first_last_touch = 0;

    //! packing kernel for contraction
    ContractionPackingTpp * m_packing;

    //! true if the threading loops have to take care of fist/last touch ops
    bool m_threading_first_last_touch = false;

    //! first/last touch type of a loop
    typedef enum {
      // no first/last touch
      NONE = 0,
      // first touch before and last touch after the loop
      BEFORE_AFTER_ITER = 1,
      // first touch before the main kernel in every iteration
      // last touch after the main kernel in very iteration
      EVERY_ITER = 2
    } touch_t;

    //! number of loops
    int64_t m_num_loops = -1;
    //! dimension types of the loops (C, M, N or K)
    std::vector< dim_t >   m_loop_dim_type;
    //! sizes of the loops / number of iterations
    std::vector< int64_t > m_loop_sizes;
    //! per-loop-iteration stride in byte w.r.t. the left tensor
    std::vector< int64_t > m_loop_strides_left;
    //! per-loop-iteration stride in byte w.r.t. the right tensor
    std::vector< int64_t > m_loop_strides_right;
    //! per-loop-iteration stride in byte w.r.t. the auxiliary output tensor
    std::vector< int64_t > m_loop_strides_out_aux;
    //! per-loop-iteration stride in byte w.r.t. the output tensor
    std::vector< int64_t > m_loop_strides_out;

    //! true if the contraction loop interface was compiled
    bool m_compiled = false;

  protected:
    //! type of the first touch kernel
    kernel_t m_ktype_first_touch = UNDEFINED_KTYPE;
    //! type of the main kernel
    kernel_t m_ktype_main = UNDEFINED_KTYPE;
    //! type of the last touch kernel
    kernel_t m_ktype_last_touch = UNDEFINED_KTYPE;

    //! true if the outermost C dimension represents the complex dimension
    bool m_cpx_outer_c = false;

    //! complex stride of the left tensor
    int64_t m_cpx_stride_in_left_bytes = 0;
    //! complex stride of the right tensor
    int64_t m_cpx_stride_in_right_bytes = 0;
    //! complex stride of the auxiliary output tensor
    int64_t m_cpx_stride_out_aux_bytes = 0;
    //! complex stride of the output tensor
    int64_t m_cpx_stride_out_bytes = 0;

  public:
    /**
     * Kernel applied to the output tensor before the contraction.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_first_touch( void const * i_out_aux,
                                     void       * io_out ) = 0;

    /**
     * Kernel applied to the output tensor after the contraction.
     *
     * @param i_out_aux pointer to a data section of the auxiliary output tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_last_touch( void const * i_out_aux,
                                    void       * io_out ) = 0;

    /**
     * Kernel called in the innermost loop.
     *
     * @param i_left pointer to a data section of the left tensor.
     * @param i_right pointer to a data section of the right tensor.
     * @param io_out pointer to a data section of the output tensor.
     **/
    virtual void kernel_main( void const * i_left,
                              void const * i_right,
                              void       * io_out ) = 0;

    /**
     * Initializes the the class.
     * 
     * Shortcuts:
     *   C: batch dimensions which appears in all tensors.
     *   M: dimensions appear in left input and output.
     *   N: dimensions appear in right input and output.
     *   K: reduction dimensions which appear in both inputs,
     *
     * @param i_num_dims_c number of C dimensions.
     * @param i_num_dims_m number of M dimensions.
     * @param i_num_dims_n number of N dimensions.
     * @param i_num_dims_k number of K dimensions.
     * @param i_dim_ids_c dimensiom ids of the C dimensions.
     * @param i_dim_ids_m dimensiom ids of the M dimensions.
     * @param i_dim_ids_n dimensiom ids of the N dimensions.
     * @param i_dim_ids_k dimensiom ids of the K dimensions.
     * @param i_sizes sizes of the dimensions
     * @param i_strides_left strides of the left input tensor.
     * @param i_strides_right strides of the right input tensor.
     * @param i_strides_out_aux strides of the auxiliary output tensor.
     * @param i_strides_out strides of the output tensor.
     * @param i_dim_type types of the dimensions
     * @param i_loop_ids the loop execution strategy 
     * @param i_num_bytes_scalar_left number of bytes per scalar in the left tensor.
     * @param i_num_bytes_scalar_right number of bytes per scalar in the right tensor.
     * @param i_num_bytes_scalar_out number of bytes per scalar in the output tensor.
     * @param i_ktype_first_touch type of the first touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     * @param i_packing packing kernel for contraction
     **/
void init( int64_t                              i_num_dims_c,
           int64_t                              i_num_dims_m,
           int64_t                              i_num_dims_n,
           int64_t                              i_num_dims_k,
           int64_t                      const * i_dim_ids_c,
           int64_t                      const * i_dim_ids_m,
           int64_t                      const * i_dim_ids_n,
           int64_t                      const * i_dim_ids_k,
           std::map< int64_t, int64_t > const * i_sizes,
           std::map< int64_t, int64_t > const * i_strides_left,
           std::map< int64_t, int64_t > const * i_strides_right,
           std::map< int64_t, int64_t > const * i_strides_out_aux,
           std::map< int64_t, int64_t > const * i_strides_out,
           std::map< int64_t, dim_t >   const * i_dim_type,
           std::vector<int64_t>               * i_loop_ids,
           int64_t                              i_num_bytes_scalar_left,
           int64_t                              i_num_bytes_scalar_right,
           int64_t                              i_num_bytes_scalar_out,
           kernel_t                             i_ktype_first_touch,
           kernel_t                             i_ktype_main,
           kernel_t                             i_ktype_last_touch,
           ContractionPackingTpp              * i_packing );

    /**
     * Compiles the contraction loop interface.
     *
     * @return SUCCESS if the compilation was successful, otherwise an appropiate error code.
     **/
    err_t compile();

    /**
     * Derives the threading data for the contraction loops.
     * Parallelizes all loops such that the targeted number of tasks is reached or
     * all parallelizable loop dimensions have been exhausted.
     *
     * @param i_num_tasks number of tasks.
     **/
    err_t threading( int64_t i_num_tasks );

    /**
     * General purpose loop implementation featuring first and last touch operations.
     * No threading is applied.
     *
     * @param i_id_task task id which is executing the loop.
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_left pointer to the left tensor's data.
     * @param i_ptr_right pointer to the right tensor's data.
     * @param i_ptr_out_aux pointer to the auxiliary output tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     * @param i_first_access true if first time accessing this data
     * @param i_last_access true if last time accessing this data
     **/
    void contract_iter( int64_t         i_id_task,
                        int64_t         i_id_loop,
                        void    const * i_ptr_left,
                        void    const * i_ptr_right,
                        void    const * i_ptr_out_aux,
                        void          * i_ptr_out,
                        bool            i_first_access,
                        bool            i_last_access );

    /**
     * General purpose loop implementation featuring first and last touch operations.
     * Applies threading for inner loops
     *
     * @param i_id_task task id which is executing the loop.
     * @param i_id_loop dimension id of the loop which is executed.
     * @param i_ptr_left pointer to the left tensor's data.
     * @param i_ptr_right pointer to the right tensor's data.
     * @param i_ptr_out_aux pointer to the auxiliary output tensor's data.
     * @param i_ptr_out pointer to the output tensor's data.
     * @param i_first_access true if first time accessing this data
     * @param i_last_access true if last time accessing this data
     **/
    void contract_iter_non_parallel( int64_t         i_id_loop,
                                     void    const * i_ptr_left,
                                     void    const * i_ptr_right,
                                     void    const * i_ptr_out_aux,
                                     void          * i_ptr_out,
                                     bool            i_first_access,
                                     bool            i_last_access );

    /**
     * Contracts the two tensors.
     *
     * @param i_tensor_left left tensor.
     * @param i_tensor_right right tensor.
     * @param i_tensor_out_aux auxiliary data w.r.t. output tensor.
     * @param io_tensor_out output tensor.
     **/
    void contract( void const * i_tensor_left,
                   void const * i_tensor_right,
                   void const * i_tensor_out_aux,
                   void       * io_tensor_out );


    /**
     * Helper function for map find with default value
     *
     * @param i_map map.
     * @param i_key key.
     * @param i_default default value.
     * 
     * @param return value or default value.
     **/
    template <typename T>
    T map_find_default( std::map< int64_t, T > const * i_map,
                        int64_t                        i_key,
                        T                              i_default){
      if(auto search = i_map->find(i_key); search != i_map->end() ) {
        
        return search->second;
      }
      else {
        return i_default;
      }
    }
};

#endif