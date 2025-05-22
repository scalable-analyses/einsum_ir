#include "EinsumTree.h"
#include "EinsumTreeAscii.h"
#ifdef _OPENMP
#include "omp.h"
#endif

void einsum_ir::frontend::EinsumTree::init( std::vector< std::vector< int64_t > >         * i_dim_ids,
                                            std::vector< std::vector< int64_t > >         * i_children,
                                            std::map < int64_t, int64_t >                 * i_map_dim_sizes,
                                            data_t                                          i_dtype,
                                            void                                  * const * i_data_ptrs ){
  
  m_dim_ids = i_dim_ids;
  m_children = i_children;
  m_map_dim_sizes = i_map_dim_sizes;
  m_dtype = i_dtype;
  m_data_ptrs = i_data_ptrs;
}

einsum_ir::err_t einsum_ir::frontend::EinsumTree::compile() {
  err_t l_err = err_t::UNDEFINED_ERROR;

#ifdef _OPENMP
  int64_t l_num_threads = omp_get_max_threads();
#else
  int64_t l_num_threads = 1;
#endif

  m_nodes.resize( m_children->size() );

  //initialize nodes
  for( size_t l_node_id = 0; l_node_id < m_nodes.size(); l_node_id++){
    std::vector< int64_t > * l_children = &m_children->at(    l_node_id );
    std::vector< int64_t > * l_dim_ids  = &m_dim_ids->at(     l_node_id );
    backend::EinsumNode    * l_node     = &m_nodes[           l_node_id ];
    void                   * l_data_ptr = m_data_ptrs[        l_node_id ];

    //leaf node
    if( l_children->size() == 0 ){
      l_node->init( l_dim_ids->size(),
                    l_dim_ids->data(),
                    m_map_dim_sizes,
                    nullptr,
                    m_dtype,
                    l_data_ptr,
                    &m_memory );
    }
    //unary node
    else if( l_children->size() == 1 ){
      int64_t l_child_left  = l_children->at(0);

      l_node->init( l_dim_ids->size(),
                    l_dim_ids->data(),
                    m_map_dim_sizes,
                    nullptr,
                    m_dtype,
                    l_data_ptr,
                    &m_nodes[l_child_left],
                    &m_memory,
                    l_num_threads );
    }
    //binary node
    else{
      int64_t l_child_left  = l_children->at(0);
      int64_t l_child_right = l_children->at(1);
      kernel_t l_ktype_first_touch = einsum_ir::ZERO;
      kernel_t l_ktype_main        = einsum_ir::MADD;

      l_node->init( l_dim_ids->size(),
                    l_dim_ids->data(),
                    m_map_dim_sizes,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    m_dtype,
                    nullptr,
                    l_data_ptr,
                    l_ktype_first_touch,
                    l_ktype_main,
                    kernel_t::UNDEFINED_KTYPE,
                    &m_nodes[l_child_left],
                    &m_nodes[l_child_right],
                    &m_memory,
                    l_num_threads );
    }
  }
  
  //compile all nodes
  l_err = m_nodes.back().compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return l_err;
  }

  return einsum_ir::SUCCESS;
}

void einsum_ir::frontend::EinsumTree::eval() {
  m_nodes.back().eval();
}

int64_t einsum_ir::frontend::EinsumTree::num_ops() {
  if( m_nodes.size() > 0 ) {
    return m_nodes.back().num_ops( true );
  }
  else {
    return 0;
  }
}