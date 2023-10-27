#include "EinsumNode.h"
#include "UnaryTpp.h"
#include "BinaryContractionTpp.h"
#include "Tensor.h"

einsum_ir::backend::EinsumNode::~EinsumNode() {
  if( m_unary != nullptr ) {
    delete m_unary;
  }
  if( m_cont != nullptr ) {
    delete m_cont;
  }
  if( m_data_ptr_int != nullptr ) {
    delete [] (char *) m_data_ptr_int;
  }
}

void einsum_ir::backend::EinsumNode::init( int64_t                              i_num_dims,
                                           int64_t                      const * i_dim_ids,
                                           std::map< int64_t, int64_t > const * i_dim_sizes_inner,
                                           std::map< int64_t, int64_t > const * i_dim_sizes_outer,
                                           data_t                               i_dtype,
                                           void                               * i_data_ptr ) {
  m_dtype               = i_dtype;

  m_ktype_first_touch   = kernel_t::UNDEFINED_KTYPE;
  m_ktype_main          = kernel_t::UNDEFINED_KTYPE;
  m_ktype_last_touch    = kernel_t::UNDEFINED_KTYPE;

  m_size                = 0;

  m_num_dims            = i_num_dims;
  m_dim_ids_ext         = i_dim_ids;
  m_dim_ids_int         = nullptr;

  m_dim_sizes_inner     = i_dim_sizes_inner;
  if( i_dim_sizes_outer != nullptr ) {
    m_dim_sizes_outer   = i_dim_sizes_outer;
  }
  else {
    m_dim_sizes_outer   = i_dim_sizes_inner;
  }

  m_offsets_aux_ext     = nullptr;
  m_offsets_ext         = nullptr;

  m_strides_children.resize(0);
  m_strides             = nullptr;

  m_dim_link_s_to_p     = nullptr;

  m_children.resize(0);
  m_data_ptr_int        = nullptr;
  m_data_ptr_ext        = i_data_ptr;

  m_cont                = nullptr;

  m_num_ops_node        = 0;
  m_num_ops_children    = 0;

  m_num_tasks_intra_op  = 1;

  m_compiled            = false;
  m_data_locked         = false;
}

void einsum_ir::backend::EinsumNode::init( int64_t                              i_num_dims,
                                           int64_t                      const * i_dim_ids,
                                           std::map< int64_t, int64_t > const * i_dim_sizes_inner,
                                           std::map< int64_t, int64_t > const * i_dim_sizes_aux_outer,
                                           std::map< int64_t, int64_t > const * i_dim_sizes_outer,
                                           std::map< int64_t, int64_t > const * i_offsets_aux,
                                           std::map< int64_t, int64_t > const * i_offsets,
                                           std::map< int64_t, int64_t > const * i_strides_left,
                                           std::map< int64_t, int64_t > const * i_strides_right,
                                           std::map< int64_t, int64_t > const * i_strides,
                                           std::map< int64_t, int64_t > const * i_dim_link_s_to_p,
                                           data_t                               i_dtype,
                                           void                               * i_data_ptr_aux,
                                           void                               * i_data_ptr,
                                           kernel_t                             i_ktype_first_touch,
                                           kernel_t                             i_ktype_main,
                                           kernel_t                             i_ktype_last_touch,
                                           EinsumNode                         * i_left,
                                           EinsumNode                         * i_right ) {
  init( i_num_dims,
        i_dim_ids,
        i_dim_sizes_inner,
        i_dim_sizes_outer,
        i_dtype,
        i_data_ptr );

  m_dim_sizes_aux_outer = i_dim_sizes_aux_outer;
  m_offsets_aux_ext     = i_offsets_aux;
  m_offsets_ext         = i_offsets;
  m_strides             = i_strides;
  m_dim_link_s_to_p     = i_dim_link_s_to_p;
  m_data_ptr_aux_ext    = i_data_ptr_aux;
  m_ktype_first_touch   = i_ktype_first_touch;
  m_ktype_main          = i_ktype_main;
  m_ktype_last_touch    = i_ktype_last_touch;

  m_strides_children.resize( 2 );
  m_strides_children[0] = i_strides_left;
  m_strides_children[1] = i_strides_right;

  m_children.resize( 2 );
  m_children[0] = i_left;
  m_children[1] = i_right;
}

einsum_ir::err_t einsum_ir::backend::EinsumNode::compile( int64_t const * i_dim_ids_req ) {
  err_t l_err = err_t::UNDEFINED_ERROR;

  m_dim_ids_int = i_dim_ids_req;

  m_size = Tensor::size( ce_n_bytes( m_dtype ),
                         m_num_dims,
                         m_dim_ids_ext,
                         *m_dim_sizes_outer );

  // compile contraction
  if( m_children.size() == 2 ) {
    m_cont = new BinaryContractionTpp;
    m_cont->init( m_children[0]->m_num_dims,
                  m_children[1]->m_num_dims,
                  m_num_dims,
                  m_dim_sizes_inner,
                  m_children[0]->m_dim_sizes_outer,
                  m_children[1]->m_dim_sizes_outer,
                  m_dim_sizes_aux_outer,
                  m_dim_sizes_outer,
                  m_strides_children[0],
                  m_strides_children[1],
                  m_strides,
                  m_children[0]->m_dim_ids_ext,
                  m_children[1]->m_dim_ids_ext,
                  i_dim_ids_req,
                  m_dim_link_s_to_p,
                  m_children[0]->m_dtype,
                  m_children[1]->m_dtype,
                  m_dtype,
                  m_dtype,
                  m_ktype_first_touch,
                  m_ktype_main,
                  m_ktype_last_touch );

    l_err = m_cont->compile();
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

  // compile unary copy operation
  m_unary = new UnaryTpp;
  m_unary->init( m_num_dims,
                 m_dim_sizes_outer,
                 m_dim_ids_ext,
                 m_dim_ids_int,
                 m_dtype,
                 m_dtype,
                 m_dtype,
                 kernel_t::COPY );

  l_err = m_unary->compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return l_err;
  }

  // allocate memory for intermediate data if required
  if( m_data_ptr_ext == nullptr || l_permute_inputs ) {
    char * l_data = new char[m_size];
    m_data_ptr_int = l_data;
  }

  // compute offsets
  if( m_offsets_aux_ext != nullptr ) {
    int64_t l_di_int = m_num_dims - 1;
    m_offset_aux_bytes = 0;
    int64_t l_stride = 1;
    while( l_di_int >= 0 ) {
      int64_t l_dim_id = m_dim_ids_int[l_di_int];
      int64_t l_offset = 0;
      if( m_offsets_aux_ext->find(l_dim_id ) != m_offsets_aux_ext->end() ) {
        l_offset = m_offsets_aux_ext->at( l_dim_id );
      }
      m_offset_bytes += l_stride * l_offset;
      l_stride *= m_dim_sizes_aux_outer->at( l_dim_id );
      l_di_int--;
    }
    m_offset_aux_bytes *= ce_n_bytes( m_dtype );
  }

  if( m_offsets_ext != nullptr ) {
    int64_t l_di_int = m_num_dims - 1;
    m_offset_bytes = 0;
    int64_t l_stride = 1;
    while( l_di_int >= 0 ) {
      int64_t l_dim_id = m_dim_ids_int[l_di_int];
      int64_t l_offset = 0;
      if( m_offsets_ext->find(l_dim_id ) != m_offsets_ext->end() ) {
        l_offset = m_offsets_ext->at( l_dim_id );
      }
      m_offset_bytes += l_stride * l_offset;
      l_stride *= m_dim_sizes_outer->at( l_dim_id );
      l_di_int--;
    }
    m_offset_bytes *= ce_n_bytes( m_dtype );
  }

  // compile children
  for( std::size_t l_ch = 0; l_ch < m_children.size(); l_ch++ ) {
    int64_t const * l_dim_ids_child = m_cont->dim_ids_in_ordered( l_ch );
    l_err = m_children[l_ch]->compile( l_dim_ids_child );
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
  m_unary->eval( m_data_ptr_ext,
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
    m_unary->eval( m_data_ptr_ext,
                   m_data_ptr_int );
  }

  if( m_children.size() == 2 ) {
    void const * l_left_int = m_children[0]->m_data_ptr_int;
    void const * l_left_ext = m_children[0]->m_data_ptr_ext;
    void const * l_left = l_left_int != nullptr ? l_left_int : l_left_ext;

    void const * l_right_int = m_children[1]->m_data_ptr_int;
    void const * l_right_ext = m_children[1]->m_data_ptr_ext;
    void const * l_right = l_right_int != nullptr ? l_right_int : l_right_ext;

    void const * l_data_aux = m_data_ptr_aux_int != nullptr ? m_data_ptr_aux_int : m_data_ptr_aux_ext;
    l_data_aux = (char *) l_data_aux + m_offset_aux_bytes;

    void * l_data = m_data_ptr_int != nullptr ? m_data_ptr_int  : m_data_ptr_ext;
    l_data = (char *) l_data + m_offset_bytes;

    m_cont->contract( l_left,
                      l_right,
                      l_data_aux,
                      l_data );
  }
}

int64_t einsum_ir::backend::EinsumNode::num_ops( bool i_children ) {
  int64_t l_num_ops = m_num_ops_node;

  if( i_children ) {
    l_num_ops += m_num_ops_children;
  }

  return l_num_ops;
}