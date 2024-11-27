#include "ContractionPackingTpp.h"
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

einsum_ir::backend::ContractionPackingTpp::~ContractionPackingTpp() {
  if( m_unary_left != nullptr ) {
    delete m_unary_left;
  }
  if( m_unary_right != nullptr ) {
    delete m_unary_right;
  }
}

void einsum_ir::backend::ContractionPackingTpp::init( int64_t                              i_num_dims_left,
                                                      int64_t                              i_num_dims_right,
                                                      std::map< int64_t, int64_t > const * i_dim_sizes,
                                                      std::map< int64_t, int64_t > const * i_strides_left,
                                                      std::map< int64_t, int64_t > const * i_strides_right,
                                                      std::map< int64_t, dim_t >   const * i_dim_type,
                                                      int64_t                      const * i_dim_ids_left,
                                                      int64_t                      const * i_dim_ids_right,
                                                      std::vector< int64_t >       const * i_dim_ids_kernel_left,
                                                      std::vector< int64_t >       const * i_dim_ids_kernel_right,
                                                      std::vector< int64_t >       const * i_loop_dims,
                                                      data_t                               i_dtype_left,
                                                      data_t                               i_dtype_right,
                                                      MemoryManager                      * i_memory ) {

  m_num_dims_left = i_num_dims_left;
  m_num_dims_right = i_num_dims_right;

  m_dim_sizes = i_dim_sizes;
  m_strides_left = i_strides_left;
  m_strides_right = i_strides_right;
  m_dim_type = i_dim_type;

  m_dim_ids_left = i_dim_ids_left;
  m_dim_ids_right = i_dim_ids_right;

  m_dim_ids_kernel_left = i_dim_ids_kernel_left;
  m_dim_ids_kernel_right = i_dim_ids_kernel_right;

  m_dtype_left = i_dtype_left;
  m_dtype_right = i_dtype_right;

  m_loop_dims = i_loop_dims;

  m_memory = i_memory;
}

einsum_ir::err_t einsum_ir::backend::ContractionPackingTpp::compile() {
  err_t l_err = err_t::UNDEFINED_ERROR;

  int64_t l_dim_id_extra_left = -1;
  int64_t l_dim_id_extra_right = -1;
  std::vector< int64_t > l_dim_ids_possible_extra;
  std::vector< int64_t > l_dim_ids_extra_packing;

  //determine if extra packing dimensions are necessary for left input
  if( m_dim_ids_kernel_left->size() > 0 ){
    int64_t l_stride_one_dim =  m_dim_ids_left[ m_num_dims_left - 1 ];
    if( std::find( m_dim_ids_kernel_left->begin(), m_dim_ids_kernel_left->end(), l_stride_one_dim ) == m_dim_ids_kernel_left->end() ) {
      l_dim_ids_possible_extra.push_back(l_stride_one_dim);
      l_dim_id_extra_left = l_stride_one_dim;
    }
  }

  //determine if extra packing dimensions are necessary for right input
  if( m_dim_ids_kernel_right->size() > 0 ){
    int64_t l_stride_one_dim =  m_dim_ids_right[ m_num_dims_right - 1 ];
    if( std::find( m_dim_ids_kernel_right->begin(), m_dim_ids_kernel_right->end(), l_stride_one_dim ) == m_dim_ids_kernel_right->end() ) {
       l_dim_ids_possible_extra.push_back(l_stride_one_dim);
       l_dim_id_extra_right = l_stride_one_dim;
    }
  }

  //check if usage of extra packing dims is possible with given loop order
  for(int64_t l_id = m_loop_dims->size()-1; l_id >= 0; l_id-- ){
    auto l_found = std::find( l_dim_ids_possible_extra.begin(), l_dim_ids_possible_extra.end(), m_loop_dims->at(l_id));
    if( l_found != l_dim_ids_possible_extra.end() ) {
      l_dim_ids_extra_packing.insert(l_dim_ids_extra_packing.begin(), *l_found);
    }
    else{
      break;
    }
  }

  //determine packing dimensions
  std::vector< int64_t > l_dim_ids_packing_left;
  std::vector< int64_t > l_dim_ids_packing_right;
  l_dim_ids_packing_left.reserve(  l_dim_ids_extra_packing.size() + m_dim_ids_kernel_left->size()  );
  l_dim_ids_packing_right.reserve( l_dim_ids_extra_packing.size() + m_dim_ids_kernel_right->size() );

  for( int64_t l_id = 0; l_id < l_dim_ids_extra_packing.size(); l_id++ ) {
    dim_t dim_type = m_dim_type->at( l_dim_ids_extra_packing[l_id] );
    if( m_packing_loop_offset_left || l_dim_ids_extra_packing[l_id] == l_dim_id_extra_left ) {
      m_packing_loop_offset_left++;
      if( dim_type != einsum_ir::N ) {
        l_dim_ids_packing_left.push_back( l_dim_ids_extra_packing[l_id] );
      }
    }
    if( m_packing_loop_offset_right || l_dim_ids_extra_packing[l_id] == l_dim_id_extra_right ) {
      m_packing_loop_offset_right++;
      if( dim_type != einsum_ir::M ){
        l_dim_ids_packing_right.push_back( l_dim_ids_extra_packing[l_id] );
      }
    }
  }

  //insert kernel dimensions
  l_dim_ids_packing_left.insert(  l_dim_ids_packing_left.end(),  m_dim_ids_kernel_left->begin(),  m_dim_ids_kernel_left->end()  );
  l_dim_ids_packing_right.insert( l_dim_ids_packing_right.end(), m_dim_ids_kernel_right->begin(), m_dim_ids_kernel_right->end() );

  //create packing kernel
  if( l_dim_ids_packing_left.size() > 0 ) {
    m_packing_loop_offset_left++;
    m_unary_left = new UnaryTpp;
    l_err = create_kernel( m_num_dims_left,
                           m_dim_ids_left,
                           &l_dim_ids_packing_left,
                           m_strides_left,
                           &m_strides_packed_left,
                           m_dim_sizes,
                           m_dtype_left,
                           &m_size_packing_left,
                           m_unary_left);
    if( l_err != einsum_ir::SUCCESS ) {
      return l_err;
    }
  }

  if( l_dim_ids_packing_right.size() > 0 ) {
    m_packing_loop_offset_right++;
    m_unary_right = new UnaryTpp;
    l_err = create_kernel( m_num_dims_right,
                           m_dim_ids_right,
                           &l_dim_ids_packing_right,
                           m_strides_right,
                           &m_strides_packed_right,
                           m_dim_sizes,
                           m_dtype_right,
                           &m_size_packing_right,
                           m_unary_right);
    if( l_err != einsum_ir::SUCCESS ) {
      return l_err;
    }
  }

  m_memory->reserve_thread_memory( m_size_packing_left + m_size_packing_right );

  return l_err;
}

einsum_ir::err_t einsum_ir::backend::ContractionPackingTpp::create_kernel( int64_t                              i_num_dims,
                                                                           int64_t                      const * i_dim_ids_original,
                                                                           std::vector< int64_t >       const * i_dim_ids_packed,
                                                                           std::map< int64_t, int64_t > const * i_strides_original,
                                                                           std::map< int64_t, int64_t >       * o_strides_packed,
                                                                           std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                           data_t                               i_dtype,
                                                                           int64_t                            * o_size_packing,
                                                                           UnaryTpp                           * o_unary )
{
  err_t l_err = err_t::UNDEFINED_ERROR;

  //determine input dims and input strides
  std::vector< int64_t > l_dim_ids_in;
  std::vector< int64_t > l_strides_in;
  l_dim_ids_in.reserve(i_dim_ids_packed->size());
  l_strides_in.reserve(i_dim_ids_packed->size());
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    if( std::find( i_dim_ids_packed->begin(), i_dim_ids_packed->end(), i_dim_ids_original[l_di] ) != i_dim_ids_packed->end() ) {
      int64_t l_dim_id = i_dim_ids_original[l_di];
      int64_t l_stride = i_strides_original->at(l_dim_id);
      l_dim_ids_in.push_back( l_dim_id );
      l_strides_in.push_back( l_stride );
    }
  }
  //set output strides
  std::vector< int64_t > l_strides_out;
  l_strides_out.resize(i_dim_ids_packed->size());
  o_unary->strides( l_strides_out.size(),
                    i_dim_sizes,
                    i_dim_ids_packed->data(),
                    l_strides_out.data());
  for( int64_t l_di = 0; l_di < l_strides_out.size(); l_di++ ) {
    std::pair< int64_t, int64_t > l_pair( i_dim_ids_packed->data()[l_di], l_strides_out[l_di] );
    o_strides_packed->insert( l_pair );
  }

  //determine required memory
  *o_size_packing = 1;
  for( int64_t l_di = 0; l_di < i_dim_ids_packed->size(); l_di++){
    *o_size_packing *= i_dim_sizes->at( (*i_dim_ids_packed)[l_di] );
  }
  *o_size_packing *= ce_n_bytes(i_dtype);

  //! initialise packing operation
  o_unary->init( i_dim_ids_packed->size(),
                 i_dim_sizes,
                 l_dim_ids_in.data(),
                 i_dim_ids_packed->data(),
                 l_strides_in.data(),
                 l_strides_out.data(),
                 i_dtype,
                 i_dtype,
                 i_dtype,
                 kernel_t::COPY );

  o_unary->threading(1);
  l_err = o_unary->compile();

  return l_err;
}

char * einsum_ir::backend::ContractionPackingTpp::kernel_pack_left( char * i_in ) {
  int64_t l_thread_num = 0;
  #ifdef _OPENMP
  l_thread_num = omp_get_thread_num();
  #endif

  m_unary_left->eval( i_in,
                      m_memory_packing[l_thread_num] );
  return m_memory_packing[l_thread_num];
}

char * einsum_ir::backend::ContractionPackingTpp::kernel_pack_right( char * i_in ) {
  int64_t l_thread_num = 0;
  #ifdef _OPENMP
  l_thread_num = omp_get_thread_num();
  #endif

  m_unary_right->eval( i_in,
                       m_memory_packing[l_thread_num] + m_size_packing_left );
  return m_memory_packing[l_thread_num] + m_size_packing_left;
}

void einsum_ir::backend::ContractionPackingTpp::allocate_memory(){
  m_memory_packing = (char**)m_memory->get_thread_memory();
}
