#include "EinsumNode.h"
#include "UnaryTpp.h"
#include "Tensor.h"
#include "BinaryContractionFactory.h"
#include "BinaryPrimitives.h"
#include <algorithm>
#include <cstdlib>

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
  m_dim_ids_int.resize( m_num_dims );
  std::copy( m_dim_ids_ext,
             m_dim_ids_ext + m_num_dims,
             m_dim_ids_int.begin() );

  m_dim_sizes_inner     = i_dim_sizes_inner;
  if( i_dim_sizes_outer != nullptr ) {
    m_dim_sizes_outer   = i_dim_sizes_outer;
  }
  else {
    m_dim_sizes_outer   = i_dim_sizes_inner;
  }

  m_offsets_aux_ext     = nullptr;
  m_offsets_ext         = nullptr;

  m_children.resize(0);
  m_data_ptr_int        = nullptr;
  m_data_ptr_ext        = i_data_ptr;

  m_btype_unary         = backend_t::AUTO;
  m_btype_binary        = backend_t::AUTO;
  char * l_btype = std::getenv( "EINSUM_IR_BACKEND" );
  if( l_btype == nullptr ) {}
  else if( strcmp( l_btype, "AUTO") == 0 ) {
    m_btype_binary = backend_t::AUTO;
  }
  else if( strcmp( l_btype, "TPP") == 0 ) {
    m_btype_binary = backend_t::TPP;
  }
  else if( strcmp( l_btype, "BLAS") == 0 ) {
    m_btype_binary = backend_t::BLAS;
  }
  else if( strcmp( l_btype, "TBLIS") == 0 ) {
    m_btype_binary = backend_t::TBLIS;
  }
  else if( strcmp( l_btype, "SCALAR") == 0 ) {
    m_btype_binary = backend_t::SCALAR;
  }

  m_reorder_dims = true;
  char * l_reorder_dims = std::getenv( "EINSUM_IR_REORDER_DIMS" );
  if( l_reorder_dims != nullptr ) {
    if( strcmp( l_reorder_dims, "1" ) == 0 ) {
      m_reorder_dims = true;
    }
    else if( strcmp( l_reorder_dims, "true" ) == 0 ) {
      m_reorder_dims = true;
    }
    else {
      m_reorder_dims = false;
    }
  }

  m_unary               = nullptr;
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
                                           std::map< int64_t, int64_t > const * i_dim_sizes_outer,
                                           data_t                               i_dtype,
                                           void                               * i_data_ptr,
                                           EinsumNode                         * i_child ) {
  init( i_num_dims,
        i_dim_ids,
        i_dim_sizes_inner,
        i_dim_sizes_outer,
        i_dtype,
        i_data_ptr );

  m_children.resize(1);
  m_children[0] = i_child;
}

void einsum_ir::backend::EinsumNode::init( int64_t                              i_num_dims,
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
  m_data_ptr_aux_ext    = i_data_ptr_aux;
  m_ktype_first_touch   = i_ktype_first_touch;
  m_ktype_main          = i_ktype_main;
  m_ktype_last_touch    = i_ktype_last_touch;

  m_children.resize( 2 );
  m_children[0] = i_left;
  m_children[1] = i_right;
}

einsum_ir::err_t einsum_ir::backend::EinsumNode::compile() {
  err_t l_err = err_t::UNDEFINED_ERROR;

  // derive backend for binary contractions
  if( m_btype_binary == backend_t::AUTO ) {
    if(    ce_cpx_op(m_ktype_first_touch)
        || ce_cpx_op(m_ktype_main)
        || ce_cpx_op(m_ktype_last_touch) ) {
      m_btype_binary = backend_t::BLAS;
    }
    else if( BinaryContractionFactory::supports( backend_t::TPP ) ) {
      m_btype_binary = backend_t::TPP;
    }
    else if( BinaryContractionFactory::supports( backend_t::TBLIS ) ) {
      m_btype_binary = backend_t::TBLIS;
    }
    else if( BinaryContractionFactory::supports( backend_t::BLAS ) ) {
      m_btype_binary = backend_t::BLAS;
    }
    else {
      m_btype_binary = backend_t::SCALAR;
    }
  }

  if( BinaryContractionFactory::supports( m_btype_binary ) == false ) {
    return err_t::INVALID_BACKEND;
  }

  m_size = Tensor::size( ce_n_bytes( m_dtype ),
                         m_num_dims,
                         m_dim_ids_ext,
                         *m_dim_sizes_outer );

  // compile contraction
  if( m_children.size() == 2 ) {
    // swap left and right if required by the primitives
    bool l_swap_inputs = BinaryPrimitives::swap_inputs( m_children[0]->m_num_dims,
                                                        m_children[1]->m_num_dims,
                                                        m_num_dims,
                                                        m_children[0]->m_dim_ids_ext,
                                                        m_children[1]->m_dim_ids_ext,
                                                        m_dim_ids_int.data() );
    if( l_swap_inputs ) {
      std::swap( m_children[0],
                 m_children[1] );
    }

    // reorder dimensions of input tensors for the primitives
    if( m_reorder_dims ) {
      BinaryPrimitives l_bin_prims;
      l_bin_prims.init( m_dtype,
                        m_btype_binary );

      l_err = l_bin_prims.reorder( m_btype_binary,
                                  m_children[0]->m_num_dims,
                                  m_children[1]->m_num_dims,
                                  m_num_dims,
                                  m_dim_sizes_inner,
                                  m_children[0]->m_dim_ids_int.data(),
                                  m_children[1]->m_dim_ids_int.data(),
                                  m_dim_ids_int.data() );
      if( l_err != einsum_ir::SUCCESS ) {
        return l_err;
      }
    }

    m_cont = BinaryContractionFactory::create( m_btype_binary );
    m_cont->init( m_children[0]->m_num_dims,
                  m_children[1]->m_num_dims,
                  m_num_dims,
                  m_dim_sizes_inner,
                  m_children[0]->m_dim_sizes_outer,
                  m_children[1]->m_dim_sizes_outer,
                  m_dim_sizes_aux_outer,
                  m_dim_sizes_outer,
                  m_children[0]->m_dim_ids_int.data(),
                  m_children[1]->m_dim_ids_int.data(),
                  m_dim_ids_int.data(),
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
  if(    m_dim_ids_ext != m_dim_ids_int.data()
      && m_data_ptr_ext != nullptr ) {
    for( int64_t l_di = 0; l_di < m_num_dims; l_di++ ) {
      if( m_dim_ids_ext[l_di] != m_dim_ids_int.data()[l_di] ) {
        l_permute_inputs = true;
        break;
      }
    }
  }

  // compile unary copy operation
  m_unary = new UnaryTpp;

  if( m_children.size() != 1 ) {
    m_unary->init( m_num_dims,
                   m_dim_sizes_outer,
                   m_dim_ids_ext,
                   m_dim_ids_int.data(),
                   m_dtype,
                   m_dtype,
                   m_dtype,
                   kernel_t::COPY );
  }
  else {
    m_unary->init( m_num_dims,
                   m_dim_sizes_outer,
                   m_children[0]->m_dim_ids_ext,
                   m_dim_ids_ext,
                   m_dtype,
                   m_dtype,
                   m_dtype,
                   kernel_t::COPY );
  }

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
  if( m_children.size() > 1 ) {
    for( std::size_t l_ch = 0; l_ch < m_children.size(); l_ch++ ) {
      l_err = m_children[l_ch]->compile();
      if( l_err != einsum_ir::SUCCESS ) {
        return l_err;
      }
    }
  }
  else if( m_children.size() == 1 ) {
    l_err = m_children[0]->compile();
    if( l_err != einsum_ir::SUCCESS ) {
      return l_err;
    }
  }

  // derive the number of ops
  m_num_ops_node = 0;
  m_num_ops_children = 0;

  if( m_children.size() > 1 ) {
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
    if( m_unary != nullptr ) m_unary->threading( m_num_tasks_intra_op );
    if( m_cont  != nullptr ) m_cont->threading(  m_num_tasks_intra_op );
  }
#endif
  return einsum_ir::SUCCESS;
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

  if( m_children.size() != 1 ) {
    if(    m_data_locked  == false
        && m_data_ptr_ext != nullptr
        && m_data_ptr_int != nullptr ) {
      m_unary->eval( m_data_ptr_ext,
                     m_data_ptr_int );
    }
  }
  else {
    if( m_children[0]->m_data_ptr_int != nullptr ) {
      m_unary->eval( m_children[0]->m_data_ptr_int,
                     m_data_ptr_ext );
    }
    else {
      m_unary->eval( m_children[0]->m_data_ptr_ext,
                     m_data_ptr_ext );
    }
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
