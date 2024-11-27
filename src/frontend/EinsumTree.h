#ifndef EINSUM_IR_FRONTEND_EINSUM_TREE
#define EINSUM_IR_FRONTEND_EINSUM_TREE

#include <string>
#include <vector>
#include <map>
#include "../backend/EinsumNode.h"

namespace einsum_ir {
  namespace frontend {
    class EinsumTree;
  }
}

class einsum_ir::frontend::EinsumTree {
  public:
    //! nodes of the resulting einsum tree
    std::vector< backend::EinsumNode > m_nodes;

    //vector with dim ids of all tensor
    std::vector< std::vector< int64_t > > * m_dim_ids;

    //vector with children of all tensor
    std::vector< std::vector< int64_t > > * m_children;

    //! data points of the tensors 
    void * const * m_data_ptrs = nullptr;

    //! Memory Manager
    einsum_ir::backend::MemoryManager m_memory;

    //! datatype of all tensors
    data_t m_dtype = data_t::UNDEFINED_DTYPE;

    //! mapping from dim ids to sizes
    std::map< int64_t, int64_t > * m_map_dim_sizes;

    /**
     * Initializes the einsum tree.
     * @param i_dim_ids vector of all tensors with their dimension ids
     * @param i_children vector of all tensors with their children
     * @param i_map_dim_sizes map of dimension ids to dimension sizes
     * @param i_dtype datatype of all tensors.
     * @param i_data_ptr pointers to the tensor's data. nullptr if the tensor does not have extrenal data
     **/
    void init( std::vector< std::vector< int64_t > >         * i_dim_ids,
               std::vector< std::vector< int64_t > >         * i_children,
               std::map < int64_t, int64_t >                 * i_map_dim_sizes,
               data_t                                          i_dtype,
               void                                  * const * i_data_ptrs );

    /**
     * Compiles the einsum tree. 
     **/
    err_t compile();

    /**
     * Evaluates the einsum tree.
     */
    void eval();

    /**
     * Gets the number of scalar operations required to evaluate the expression.
     *
     * @return number of scalar operations.
     **/
    int64_t num_ops();


};

#endif
