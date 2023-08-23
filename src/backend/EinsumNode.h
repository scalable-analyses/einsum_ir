#ifndef EINSUM_IR_BACKEND_EINSUM_NODE
#define EINSUM_IR_BACKEND_EINSUM_NODE

#include <vector>
#include "BinaryContraction.h"
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
    //! type of the inner kernel
    kernel_t m_ktype_inner = kernel_t::UNDEFINED_KTYPE;
    //! type of the last-touch kernel
    kernel_t m_ktype_last_touch = kernel_t::UNDEFINED_KTYPE;

    //! size of the node's tensor in bytes
    int64_t m_size = 0;

    //! number of dimensions of the node's tensor
    int64_t m_num_dims = 0;
    //! external dimension ids of the node's tensor
    int64_t const * m_dim_ids_ext = nullptr;
    //! internal dimension ids of the node's tensor
    int64_t const * m_dim_ids_int = nullptr;

    //! id to size mapping for the dimensions
    std::map< int64_t, int64_t > const * m_dim_sizes;

    //! children of the node
    std::vector< EinsumNode * > m_children;
    //! internal data
    void * m_data_ptr_int = nullptr;
    //! external data
    void * m_data_ptr_ext = nullptr;

    //! binary contraction
    BinaryContraction * m_cont = nullptr;

    //! number of operations in the contraction
    int64_t m_num_ops_node = 0;
    //! number of operations of the children
    int64_t m_num_ops_children = 0;

    //! number of intra-op tasks
    int64_t m_num_tasks_intra_op = 1;

    /**
     * Destructor.
     **/
    ~EinsumNode();

    /**
     * Initializes the node without any children and with a data pointer.
     *
     * @param i_num_dims number of tensor dimensions.
     * @param i_dim_ids ids of the tensor dimensions.
     * @param i_dim_sizes dimension id to size mapping.
     * @param i_dtype datatype of the tensor.
     * @param i_data_ptr data pointer of the tensor.
     **/
    void init( int64_t                              i_num_dims,
               int64_t                      const * i_dim_ids,
               std::map< int64_t, int64_t > const & i_dim_sizes,
               data_t                               i_dtype,
               void                               * i_data_ptr );

    /**
     * Initializes the node with two children and without a data pointer.
     *
     * @param i_num_dims number of tensor dimensions.
     * @param i_dim_ids ids of the tensor dimensions.
     * @param i_dtype datatype of the node's internal data.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_inner type of the inner kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     * @param i_left left child.
     * @param i_right right child. 
     **/
    void init( int64_t            i_num_dims,
               int64_t    const * i_dim_ids,
               data_t             i_dtype,
               kernel_t           i_ktype_first_touch,
               kernel_t           i_ktype_inner,
               kernel_t           i_ktype_last_touch,
               EinsumNode       & i_left,
               EinsumNode       & i_right );

    /**
     * Initializes the node with two children and a data pointer.
     *
     * @param i_num_dims number of tensor dimensions.
     * @param i_dim_ids ids of the tensor dimensions.
     * @param i_dtype datatype of the node's tensor.
     * @param i_data_ptr data pointer of the tensor.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_inner type of the inner kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     * @param i_left left child.
     * @param i_right right child. 
     **/
    void init( int64_t            i_num_dims,
               int64_t    const * i_dim_ids,
               data_t             i_dtype,
               void             * i_data_ptr,
               kernel_t           i_ktype_first_touch,
               kernel_t           i_ktype_inner,
               kernel_t           i_ktype_last_touch,
               EinsumNode       & i_left,
               EinsumNode       & i_right );

    /**
     * Compiles the contraction of the node and recursively those of all children.
     *
     * @param i_dim_ids_req required ordering of the tensors dimension. 
     **/
    err_t compile( int64_t const * i_dim_ids_req );

    /**
     * Compiles the contraction of the node and recursively those of all children.
     **/    
    err_t compile();

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
};

#endif