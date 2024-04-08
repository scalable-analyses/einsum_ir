#ifndef EINSUM_IR_BACKEND_EINSUM_NODE
#define EINSUM_IR_BACKEND_EINSUM_NODE

#include <vector>
#include "Unary.h"
#include "BinaryContraction.h"
#include "MemoryManager.h"
#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class EinsumNode;
  }
}

class einsum_ir::backend::EinsumNode {
  public:
    //! data type of the tensor
    data_t m_dtype = data_t::UNDEFINED_DTYPE;

    //! type of the first-touch kernel
    kernel_t m_ktype_first_touch = kernel_t::UNDEFINED_KTYPE;
    //! type of the main kernel
    kernel_t m_ktype_main = kernel_t::UNDEFINED_KTYPE;
    //! type of the last-touch kernel
    kernel_t m_ktype_last_touch = kernel_t::UNDEFINED_KTYPE;

    //! size of the node's tensor in bytes
    int64_t m_size = 0;

    //! number of dimensions of the node's tensor
    int64_t m_num_dims = 0;
    //! external dimension ids of the node's tensor
    int64_t const * m_dim_ids_ext = nullptr;
    //! internal dimension ids of the node's tensor
    std::vector< int64_t > m_dim_ids_int;

    //! id to size mapping for the inner dimensions
    std::map< int64_t, int64_t > const * m_dim_sizes_inner = nullptr;

    //! id to size mapping for the auxiliary tensor's (if any) outer dimensions
    std::map< int64_t, int64_t > const * m_dim_sizes_aux_outer = nullptr;

    //! id to size mapping for the tensor's (if any) outer dimensions
    std::map< int64_t, int64_t > const * m_dim_sizes_outer = nullptr;


    //! children of the node
    std::vector< EinsumNode * > m_children;
    //! internal data
    void * m_data_ptr_int = nullptr;
    //! external data
    void * m_data_ptr_ext = nullptr;

    //! internal auxiliary data
    void * m_data_ptr_aux_int = nullptr;
    //! external auxiliary data
    void * m_data_ptr_aux_ext = nullptr;

    //! external local auxiliary tensor offset in bytes
    std::map< int64_t, int64_t > const * m_offsets_aux_ext = nullptr;
    //! external local tensor offset in bytes
    std::map< int64_t, int64_t > const * m_offsets_ext = nullptr;

    //! effective auxiliary offset in bytes
    int64_t m_offset_aux_bytes = 0;
    //! effective offset in bytes
    int64_t m_offset_bytes = 0;

    //! true if dimension reordering is enabled
    bool m_reorder_dims = false;

    //! backend types
    backend_t m_btype_unary  = backend_t::UNDEFINED_BACKEND;
    backend_t m_btype_binary = backend_t::UNDEFINED_BACKEND;

    //! unary operation
    Unary * m_unary = nullptr;

    //! binary contraction
    BinaryContraction * m_cont = nullptr;

    //! Memory manager for intermendiate results
    MemoryManager * m_memory = nullptr;

    //! id of allocated memoy
    int64_t m_mem_id = 0;

    //! the required number of frees until reserved memory is actualy freed
    int64_t m_req_men_frees = 1;

    //! the number of reserved memory frees 
    int64_t m_num_men_frees = 0;

    //! number of operations in the contraction
    int64_t m_num_ops_node = 0;
    //! number of operations of the children
    int64_t m_num_ops_children = 0;

    //! number of intra-op tasks
    int64_t m_num_tasks_intra_op = 1;

    //! true if the node has been compiled
    bool m_compiled = false;

    //! true if the external data was copied and locked
    bool m_data_locked = false;

    /**
     * Destructor.
     **/
    ~EinsumNode();

    /**
     * Initializes an input node.
     *
     * @param i_num_dims number of tensor dimensions.
     * @param i_dim_ids ids of the tensor dimensions.
     * @param i_dim_sizes_inner dimension id to inner size mapping.
     * @param i_dim_sizes_outer dimension id to outer size mapping. optional: use nullptr if not needed.
     * @param i_dtype datatype of the tensor.
     * @param i_data_ptr data pointer of the tensor.
     **/
    void init( int64_t                              i_num_dims,
               int64_t                      const * i_dim_ids,
               std::map< int64_t, int64_t > const * i_dim_sizes_inner,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer,
               data_t                               i_dtype,
               void                               * i_data_ptr,
               MemoryManager                      * i_memory );

    /**
     * Initializes a node with a single child.
     *
     * @param i_num_dims number of tensor dimensions.
     * @param i_dim_ids ids of the tensor dimensions.
     * @param i_dim_sizes_inner dimension id to inner size mapping.
     * @param i_dim_sizes_outer dimension id to outer size mapping. optional: use nullptr if not needed.
     * @param i_dtype datatype of the node's tensor.
     * @param i_data_ptr data pointer of the tensor.
     * @param i_child child of the node.
     **/
    void init( int64_t                              i_num_dims,
               int64_t                      const * i_dim_ids,
               std::map< int64_t, int64_t > const * i_dim_sizes_inner,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer,
               data_t                               i_dtype,
               void                               * i_data_ptr,
               EinsumNode                         * i_child,
               MemoryManager                      * i_memory );

    /**
     * Initializes the node with two children.
     *
     * @param i_num_dims number of tensor dimensions.
     * @param i_dim_ids ids of the tensor dimensions.
     * @param i_dim_sizes_aux_outer dimension id to outer size mapping for the auxiliary data. optional: use nullptr if not needed.
     * @param i_dim_sizes_outer dimension id to outer size mapping for the data. optional: use nullptr if not needed.
     * @param i_offsets_aux offsets applied to the auxiliary data pointer in the node-local binary contraction. optional: use nullptr if not needed.
     * @param i_offsets offsets applied to the data pointer in the node-local binary contraction. optional: use nullptr if not needed.
     * @param i_dtype datatype of the node's tensor.
     * @param i_data_ptr_aux data pointer of the auxiliary tensor. optional: use nullptr if not needed.
     * @param i_data_ptr data pointer of the tensor.
     * @param i_ktype_first_touch type of the first-touch kernel. optional: use 0 if not needed.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     * @param i_left left child.
     * @param i_right right child. 
     **/
    void init( int64_t                              i_num_dims,
               int64_t                      const * i_dim_ids,
               std::map< int64_t, int64_t > const * i_dim_sizes_inner,
               std::map< int64_t, int64_t > const * i_dim_sizes_aux_outer,
               std::map< int64_t, int64_t > const * i_dim_sizes_outer,
               std::map< int64_t, int64_t > const * i_offsets_aux,
               std::map< int64_t, int64_t > const * i_offsets,
               data_t                               i_dtype,
               void                               * i_data_ptr_aux,
               void                               * i_data_ptr,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch,
               EinsumNode                         * i_left,
               EinsumNode                         * i_right,
               MemoryManager                      * i_memory );

    /**
     * Compiles the contraction of the node and recursively those of all children.
     **/    
    err_t compile();

    /**
     * Stores the provided data internally and locks it, i.e.,
     * the provided data pointer is ignored in future evaluations.
     * Has to be called after compilation.
     **/
    err_t store_and_lock_data();

    /**
     * Unlocks the data, i.e., the provided data pointer is used
     * in future evaluations.
     **/
    err_t unlock_data();

    /**
     * Enables intra-op threading with the given number of tasks.
     *
     * @param i_num_tasks number of targeted tasks.
     **/
    err_t threading_intra_op( int64_t i_num_tasks );

    /**
     * Evaluates the einsum tree described by the node all its children. 
     **/
    void eval();

    /**
     * Gets the number of operations required to evaluate the node.
     *
     * @param i_children if true the ops include those of all nodes in the tree; otherwise only this node.
     **/
    int64_t num_ops( bool i_children = true );

    /** 
     * Cancels a memory resrevation if this method is called from all parent einsum nodes 
     **/
    void cancel_memory_reservation();
};

#endif