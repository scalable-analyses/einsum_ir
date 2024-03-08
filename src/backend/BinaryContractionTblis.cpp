#include "BinaryContractionTblis.h"
#include <set>

einsum_ir::err_t einsum_ir::backend::BinaryContractionTblis::compile() {
  BinaryContraction::compile_base();

  // abort if auxiliary output tensor is used
  if( m_dim_sizes_outer_out_aux != nullptr ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  // check that the same data type is used everywhere
  if(    m_dtype_comp != m_dtype_left
      || m_dtype_comp != m_dtype_right
      || m_dtype_comp != m_dtype_out ) {
    return einsum_ir::err_t::INVALID_DTYPE;
  }

  // check supported kernel types
  if(    m_ktype_first_touch != einsum_ir::kernel_t::UNDEFINED_KTYPE
      && m_ktype_first_touch != einsum_ir::kernel_t::ZERO ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }
  if( m_ktype_main != einsum_ir::kernel_t::MADD ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }
  if( m_ktype_last_touch != einsum_ir::kernel_t::UNDEFINED_KTYPE ) {
    return einsum_ir::err_t::INVALID_KTYPE;
  }

  int64_t const * l_dim_ids_left  = m_dim_ids_left_native;
  int64_t const * l_dim_ids_right = m_dim_ids_right_native;

  // derive strides
  std::map< int64_t, int64_t > l_strides_left;
  strides( m_num_dims_left,
           l_dim_ids_left,
           m_dim_sizes_outer_left,
           &l_strides_left );

  std::map< int64_t, int64_t > l_strides_right;
  strides( m_num_dims_right,
           l_dim_ids_right,
           m_dim_sizes_outer_right,
           &l_strides_right );

  std::map< int64_t, int64_t > l_strides_out;
  strides( m_num_dims_out,
           m_dim_ids_out,
           m_dim_sizes_outer_out,
           &l_strides_out );

  // convert strides to tblis format
  m_tblis_strides_left.resize( m_num_dims_left );
  for( int64_t l_di = 0; l_di < m_num_dims_left; l_di++ ) {
    int64_t l_id = l_dim_ids_left[ l_di ];
    m_tblis_strides_left[ l_di ] = l_strides_left.at( l_id );
  }

  m_tblis_strides_right.resize( m_num_dims_right );
  for( int64_t l_di = 0; l_di < m_num_dims_right; l_di++ ) {
    int64_t l_id = l_dim_ids_right[ l_di ];
    m_tblis_strides_right[ l_di ] = l_strides_right.at( l_id );
  }

  m_tblis_strides_out.resize( m_num_dims_out );
  for( int64_t l_di = 0; l_di < m_num_dims_out; l_di++ ) {
    int64_t l_id = m_dim_ids_out[ l_di ];
    m_tblis_strides_out[ l_di ] = l_strides_out.at( l_id );
  }

  // convert sizes to tblis format
  m_tblis_sizes_left.resize( m_num_dims_left );
  for( int64_t l_di = 0; l_di < m_num_dims_left; l_di++ ) {
    int64_t l_id = l_dim_ids_left[ l_di ];
    m_tblis_sizes_left[ l_di ] = m_dim_sizes_inner->at( l_id );
  }

  m_tblis_sizes_right.resize( m_num_dims_right );
  for( int64_t l_di = 0; l_di < m_num_dims_right; l_di++ ) {
    int64_t l_id = l_dim_ids_right[ l_di ];
    m_tblis_sizes_right[ l_di ] = m_dim_sizes_inner->at( l_id );
  }

  m_tblis_sizes_out.resize( m_num_dims_out );
  for( int64_t l_di = 0; l_di < m_num_dims_out; l_di++ ) {
    int64_t l_id = m_dim_ids_out[ l_di ];
    m_tblis_sizes_out[ l_di ] = m_dim_sizes_inner->at( l_id );
  }

  // create tensor descriptors
  if( m_dtype_comp == einsum_ir::data_t::FP32 ) {
    tblis::tblis_init_tensor_s( &m_tblis_tensor_left,
                                m_num_dims_left,
                                m_tblis_sizes_left.data(),
                                nullptr,
                                m_tblis_strides_left.data() );

    tblis::tblis_init_tensor_s( &m_tblis_tensor_right,
                                m_num_dims_right,
                                m_tblis_sizes_right.data(),
                                nullptr,
                                m_tblis_strides_right.data() );

    tblis::tblis_init_tensor_s( &m_tblis_tensor_out,
                                m_num_dims_out,
                                m_tblis_sizes_out.data(),
                                nullptr,
                                m_tblis_strides_out.data() );
  }
  else if( m_dtype_comp == einsum_ir::data_t::FP64 ) {
    tblis::tblis_init_tensor_d( &m_tblis_tensor_left,
                                m_num_dims_left,
                                m_tblis_sizes_left.data(),
                                nullptr,
                                m_tblis_strides_left.data() );

    tblis::tblis_init_tensor_d( &m_tblis_tensor_right,
                                m_num_dims_right,
                                m_tblis_sizes_right.data(),
                                nullptr,
                                m_tblis_strides_right.data() );

    tblis::tblis_init_tensor_d( &m_tblis_tensor_out,
                                m_num_dims_out,
                                m_tblis_sizes_out.data(),
                                nullptr,
                                m_tblis_strides_out.data() );
  }
  else {
    return einsum_ir::err_t::INVALID_DTYPE;
  }

  // derive unique dimension ids
  std::set< int64_t > l_ids_unique;
  for( int64_t l_di = 0; l_di < m_num_dims_left; l_di++ ) {
    l_ids_unique.insert( l_dim_ids_left[ l_di ] );
  }
  for( int64_t l_di = 0; l_di < m_num_dims_right; l_di++ ) {
    l_ids_unique.insert( l_dim_ids_right[ l_di ] );
  }
  for( int64_t l_di = 0; l_di < m_num_dims_out; l_di++ ) {
    l_ids_unique.insert( m_dim_ids_out[ l_di ] );
  }

  // create mapping to local tblis ids
  std::map< int64_t, int64_t > l_ids_map;
  int64_t l_id_tblis = 0;
  for( int64_t l_id : l_ids_unique ) {
    l_ids_map[ l_id ] = l_id_tblis;
    l_id_tblis++;
  }
  
  // create tblis ids
  m_tblis_dim_ids_left.resize( m_num_dims_left );
  for( int64_t l_di = 0; l_di < m_num_dims_left; l_di++ ) {
    int64_t l_id = l_dim_ids_left[ l_di ];
    l_id_tblis = l_ids_map[ l_id];

    if( l_id_tblis >= 256 ) {
      return einsum_ir::err_t::COMPILATION_FAILED;
    }
    m_tblis_dim_ids_left[ l_di ] = l_id_tblis;
  }

  m_tblis_dim_ids_right.resize( m_num_dims_right );
  for( int64_t l_di = 0; l_di < m_num_dims_right; l_di++ ) {
    int64_t l_id = l_dim_ids_right[ l_di ];
    l_id_tblis = l_ids_map[ l_id ];

    if( l_id_tblis >= 256 ) {
      return einsum_ir::err_t::COMPILATION_FAILED;
    }
    m_tblis_dim_ids_right[ l_di ] = l_id_tblis;
  }

  m_tblis_dim_ids_out.resize( m_num_dims_out );
  for( int64_t l_di = 0; l_di < m_num_dims_out; l_di++ ) {
    int64_t l_id = m_dim_ids_out[ l_di ];
    l_id_tblis = l_ids_map[ l_id ];

    if( l_id_tblis >= 256 ) {
      return einsum_ir::err_t::COMPILATION_FAILED;
    }
    m_tblis_dim_ids_out[ l_di ] = l_id_tblis;
  }

  return einsum_ir::SUCCESS;
}

void einsum_ir::backend::BinaryContractionTblis::threading( int64_t  ) {
  // nothing to do: tasking is handled by tblis
}

void einsum_ir::backend::BinaryContractionTblis::contract( void const * i_tensor_left,
                                                           void const * i_tensor_right,
                                                           void const *,
                                                           void       * io_tensor_out ) {
  m_tblis_tensor_left.data  = (void *) i_tensor_left;
  m_tblis_tensor_right.data = (void *) i_tensor_right;
  m_tblis_tensor_out.data   =          io_tensor_out;

  // set zero beta before contraction based on first_touch type since TBLIS resets it to 1:
  // https://github.com/devinamatthews/tblis/blob/4de1919dfe194f5e47dfc93660ce4206d8e12c4e/src/iface/3t/mult.cxx#L162
  if( m_ktype_first_touch == einsum_ir::kernel_t::ZERO ) {
    m_tblis_tensor_out.scalar = 0.0;
  }

  tblis::tblis_tensor_mult( NULL,
                            NULL,
                            &m_tblis_tensor_left,
                            m_tblis_dim_ids_left.data(),
                            &m_tblis_tensor_right,
                            m_tblis_dim_ids_right.data(),
                            &m_tblis_tensor_out,
                            m_tblis_dim_ids_out.data() );
}

void einsum_ir::backend::BinaryContractionTblis::contract( void const * i_tensor_left,
                                                           void const * i_tensor_right,
                                                           void       * io_tensor_out ) {
  contract( i_tensor_left,
            i_tensor_right,
            nullptr,
            io_tensor_out );
}