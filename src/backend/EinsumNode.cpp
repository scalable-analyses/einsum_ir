#include "EinsumNode.h"
#include "BinaryContractionTpp.h"
#include "Tensor.h"

einsum_ir::backend::EinsumNode::~EinsumNode() {
  if( m_cont != nullptr ) {
    delete m_cont;
  }
  if( m_data_ptr_int != nullptr ) {
    delete [] (char *) m_data_ptr_int;
  }
}

void einsum_ir::backend::EinsumNode::init( int64_t                              i_num_dims,
                                           int64_t                      const * i_dim_ids,
                                           std::map< int64_t, int64_t > const & i_dim_sizes,
                                           data_t                               i_dtype,
                                           void                               * i_data_ptr ) {
  m_num_dims = i_num_dims;
  m_dim_ids_ext = i_dim_ids;
  m_dim_sizes = &i_dim_sizes;
  m_dtype = i_dtype;
  m_data_ptr_ext = i_data_ptr;
  m_num_ops_node = 0;
  m_num_ops_children = 0;
  m_num_tasks_intra_op = 1;
  m_compiled = false;
  m_data_locked = false;
}

void einsum_ir::backend::EinsumNode::init( int64_t            i_num_dims,
                                           int64_t    const * i_dim_ids,
                                           data_t             i_dtype,
                                           kernel_t           i_ktype_first_touch,
                                           kernel_t           i_ktype_main,
                                           kernel_t           i_ktype_last_touch,
                                           EinsumNode       & i_left,
                                           EinsumNode       & i_right ) {
  init( i_num_dims,
        i_dim_ids,
        *i_left.m_dim_sizes,
        i_dtype,
        nullptr );

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main = i_ktype_main;
  m_ktype_last_touch = i_ktype_last_touch;

  m_children.resize( 2 );
  m_children[0] = &i_left;
  m_children[1] = &i_right;
}

void einsum_ir::backend::EinsumNode::init( int64_t            i_num_dims,
                                           int64_t    const * i_dim_ids,
                                           data_t             i_dtype,
                                           void             * i_data_ptr,
                                           kernel_t           i_ktype_first_touch,
                                           kernel_t           i_ktype_main,
                                           kernel_t           i_ktype_last_touch,
                                           EinsumNode       & i_left,
                                           EinsumNode       & i_right ) {
  init( i_num_dims,
        i_dim_ids,
        *i_left.m_dim_sizes,
        i_dtype,
        i_data_ptr );

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main = i_ktype_main;
  m_ktype_last_touch = i_ktype_last_touch;

  m_children.resize( 2 );
  m_children[0] = &i_left;
  m_children[1] = &i_right;
}

einsum_ir::err_t einsum_ir::backend::EinsumNode::compile( int64_t const * i_dim_ids_req ) {
  m_dim_ids_int = i_dim_ids_req;

  m_size = Tensor::size( ce_n_bytes( m_dtype ),
                         m_num_dims,
                         m_dim_ids_ext,
                         *m_dim_sizes );

  // compile contraction
  if( m_children.size() == 2 ) {
    m_cont = new BinaryContractionTpp;
    m_cont->init( m_children[0]->m_num_dims,
                  m_children[1]->m_num_dims,
                  m_num_dims,
                  *m_dim_sizes,
                  m_children[0]->m_dim_ids_ext,
                  m_children[1]->m_dim_ids_ext,
                  i_dim_ids_req,
                  m_children[0]->m_dtype,
                  m_children[1]->m_dtype,
                  m_dtype,
                  m_dtype,
                  m_ktype_first_touch,
                  m_ktype_main,
                  m_ktype_last_touch );

    einsum_ir::err_t l_err = m_cont->compile();
    if( l_err != einsum_ir::SUCCESS ) {
      return l_err;
    }
  }

  // check if a permutation of the inputs is required
  bool l_permute_inputs = false;
  if(    m_dim_ids_ext != m_dim_ids_int
      && m_data_ptr_ext != nullptr ) {
    for( int64_t l_di = 0; l_di < m_num_dims; l_di++ ) {
      if( m_dim_ids_ext[l_di] != m_dim_ids_int[l_di] ) {
        l_permute_inputs = true;
        break;
      }
    }
  }

  // allocate memory for intermediate data if required
  if( m_data_ptr_ext == nullptr || l_permute_inputs ) {
    char * l_data = new char[m_size];
    m_data_ptr_int = l_data;
  }

  // compile children
  for( std::size_t l_ch = 0; l_ch < m_children.size(); l_ch++ ) {
    int64_t const * l_dim_ids_child = m_cont->dim_ids_in_ordered( l_ch );
    err_t l_err = m_children[l_ch]->compile( l_dim_ids_child );
    if( l_err != einsum_ir::SUCCESS ) {
      return l_err;
    }
  }

  // derive the number of ops
  m_num_ops_node = 0;
  m_num_ops_children = 0;

  if( m_children.size() > 0 ) {
    m_num_ops_node += m_cont->num_ops();
  }
  for( std::size_t l_ch = 0; l_ch < m_children.size(); l_ch++ ) {
    m_num_ops_children += m_children[l_ch]->m_num_ops_node;
    m_num_ops_children += m_children[l_ch]->m_num_ops_children;
  }

  m_compiled = true;

  return einsum_ir::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::EinsumNode::threading_intra_op( int64_t i_num_tasks ) {
  m_num_tasks_intra_op = i_num_tasks;

#ifdef _OPENMP
  if( m_num_tasks_intra_op > 1 ) {
    m_cont->threading( m_num_tasks_intra_op );
  }
#endif
  return einsum_ir::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::EinsumNode::compile() {
  err_t l_err = compile( m_dim_ids_ext );
  return l_err;
}

einsum_ir::err_t einsum_ir::backend::EinsumNode::store_and_lock_data() {
  if( m_compiled == false ) {
    return err_t::CALLED_BEFORE_COMPILATION;
  }
  else if( m_data_ptr_ext == nullptr ) {
    return err_t::NO_DATA_PTR_PROVIDED;
  }

  // allocate memory for intermediate data if required
  if( m_data_ptr_int == nullptr ) {
    char * l_data = new char[m_size];
    m_data_ptr_int = l_data;
  }

  // store data internally
  Tensor::permute( m_num_dims,
                   *m_dim_sizes,
                   m_dim_ids_ext,
                   m_dim_ids_int,
                   m_dtype,
                   m_dtype,
                   m_data_ptr_ext,
                   m_data_ptr_int );

  m_data_locked = true;

  return err_t::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::EinsumNode::unlock_data() {
  if( m_data_ptr_ext == nullptr ) {
    return err_t::NO_DATA_PTR_PROVIDED;
  }

  m_data_locked = false;

  return err_t::SUCCESS;
}

void einsum_ir::backend::EinsumNode::eval() {
  for( std::size_t l_ch = 0; l_ch < m_children.size(); l_ch++ ) {
    m_children[l_ch]->eval();
  }

  if(    m_data_locked  == false
      && m_data_ptr_ext != nullptr
      && m_data_ptr_int != nullptr ) {
    Tensor::permute( m_num_dims,
                     *m_dim_sizes,
                     m_dim_ids_ext,
                     m_dim_ids_int,
                     m_dtype,
                     m_dtype,
                     m_data_ptr_ext,
                     m_data_ptr_int );
  }

  if( m_children.size() == 2 ) {
    void const * l_left_int = m_children[0]->m_data_ptr_int;
    void const * l_left_ext = m_children[0]->m_data_ptr_ext;

    void const * l_right_int = m_children[1]->m_data_ptr_int;
    void const * l_right_ext = m_children[1]->m_data_ptr_ext;

    m_cont->contract( l_left_int     != nullptr ? l_left_int      : l_left_ext,
                      l_right_int    != nullptr ? l_right_int     : l_right_ext,
                      m_data_ptr_int != nullptr ? m_data_ptr_int  : m_data_ptr_ext );
  }
}

int64_t einsum_ir::backend::EinsumNode::num_ops( bool i_children ) {
  int64_t l_num_ops = m_num_ops_node;

  if( i_children ) {
    l_num_ops += m_num_ops_children;
  }

  return l_num_ops;
}