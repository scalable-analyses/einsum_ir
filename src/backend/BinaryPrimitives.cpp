#include "BinaryPrimitives.h"
#include "BinaryContraction.h"

#include <algorithm>

void einsum_ir::backend::BinaryPrimitives::init( int64_t i_size_cb_min,
                                                 int64_t i_size_cb_max,
                                                 int64_t i_size_mb_min,
                                                 int64_t i_size_mb_max,
                                                 int64_t i_size_nb_min,
                                                 int64_t i_size_nb_max,
                                                 int64_t i_size_kb_min,
                                                 int64_t i_size_kb_max ) {
  m_size_cb_min = i_size_cb_min;
  m_size_cb_max = i_size_cb_max;
  m_size_mb_min = i_size_mb_min;
  m_size_mb_max = i_size_mb_max;
  m_size_nb_min = i_size_nb_min;
  m_size_nb_max = i_size_nb_max;
  m_size_kb_min = i_size_kb_min;
  m_size_kb_max = i_size_kb_max;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::init( data_t    i_data_type,
                                                             backend_t i_backend_type ) {
  if( i_backend_type == backend_t::TPP ) {
    if( i_data_type == data_t::FP32 ) {
      init(  4,  16,
            32, 128,
            12,  64,
            32, 512 );
    }
    else if( i_data_type == data_t::FP64 ){
      init(  2,   8,
            16,  64,
             6,  32,
            16, 256 );
    }
    else {
      return err_t::INVALID_DTYPE;
    }
  }
  // BLAS
  else if( i_backend_type == backend_t::BLAS ) {
    if( i_data_type == data_t::FP32 ) {
      init(  4,  16,
            32, 512,
            32, 512,
            32, 512 );
    }
    else if( i_data_type == data_t::FP64 ) {
      init(  2,   8,
            16, 256,
            16, 256,
            16, 256 );
    }
    else {
      return err_t::INVALID_DTYPE;
    }
  }
  // TBLIS
  else if( i_backend_type == backend_t::TBLIS ) {
    if( i_data_type == data_t::FP32 ) {
      init(  4,  16,
            32, 512,
            32, 512,
            32, 512 );
    }
    else if( i_data_type == data_t::FP64 ) {
      init(  2,   8,
            16, 256,
            16, 256,
            16, 256 );
    }
    else {
      return err_t::INVALID_DTYPE;
    }
  }
  else {
    return err_t::INVALID_BACKEND;
  }

  return err_t::SUCCESS;
}

bool einsum_ir::backend::BinaryPrimitives::swap_inputs( int64_t                              i_num_dims_left,
                                                        int64_t                              i_num_dims_right,
                                                        int64_t                              i_num_dims_out,
                                                        int64_t                      const * i_dim_ids_left,
                                                        int64_t                      const * i_dim_ids_right,
                                                        int64_t                      const * i_dim_ids_out ) {
  // determine dimension types of the tensors
  std::vector< einsum_ir::dim_t > l_dim_types_left;
  std::vector< einsum_ir::dim_t > l_dim_types_right;
  std::vector< einsum_ir::dim_t > l_dim_types_out;
  BinaryContraction::dim_types( i_num_dims_left,
                                i_num_dims_right,
                                i_num_dims_out,
                                i_dim_ids_left,
                                i_dim_ids_right,
                                i_dim_ids_out,
                                &l_dim_types_left,
                                &l_dim_types_right,
                                &l_dim_types_out );

  bool l_swap_inputs = false;

  // fastest output dimension is N: swap required
  if(    l_dim_types_out.size() > 0
      && l_dim_types_out.back() == N  ) {
    l_swap_inputs = true;
  }
  // fastest output dimension is C: swap depends on first non-C dimension
  else if(    l_dim_types_out.size() > 0
           && l_dim_types_out.back() == C ) {
    int64_t l_id_out = l_dim_types_out.size() - 1;

    while( l_id_out >= 0 ) {
      // no swap if first non-C dim is M
      if( l_dim_types_out[ l_id_out ] == M ) {
        break;
      }
      // swap if first non-C is N
      else if( l_dim_types_out[ l_id_out ] == N ) {
        l_swap_inputs = true;
        break;
      }
      l_id_out--;
    }
  }

  return l_swap_inputs;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::blocking_left_kb_x_mb_cb_right_nb_x_kb_cb_out_nb_x_mb_cb( int64_t                              i_size_mb_min,
                                                                                                                 int64_t                              i_size_mb_max,
                                                                                                                 int64_t                              i_size_nb_min,
                                                                                                                 int64_t                              i_size_nb_max,
                                                                                                                 int64_t                              i_size_kb_min,
                                                                                                                 int64_t                              i_size_kb_max,
                                                                                                                 int64_t                              i_num_dims_left,
                                                                                                                 int64_t                              i_num_dims_right,
                                                                                                                 int64_t                              i_num_dims_out,
                                                                                                                 int64_t                      const * i_dim_ids_left,
                                                                                                                 int64_t                      const * i_dim_ids_right,
                                                                                                                 int64_t                      const * i_dim_ids_out,
                                                                                                                 std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                                                                 std::map< int64_t, int64_t > const * i_strides_left,
                                                                                                                 std::map< int64_t, int64_t > const * i_strides_right,
                                                                                                                 std::map< int64_t, int64_t > const * i_strides_out,
                                                                                                                 std::vector< int64_t >             * o_dim_ids_cb,
                                                                                                                 std::vector< int64_t >             * o_dim_ids_mb,
                                                                                                                 std::vector< int64_t >             * o_dim_ids_nb,
                                                                                                                 std::vector< int64_t >             * o_dim_ids_kb ) {
  int64_t l_size_cb = 1;
  int64_t l_size_mb = 1;
  int64_t l_size_nb = 1;
  int64_t l_size_kb = 1;

  // init output
  o_dim_ids_cb->resize( 0 );
  o_dim_ids_mb->resize( 0 );
  o_dim_ids_nb->resize( 0 );
  o_dim_ids_kb->resize( 0 );

  // determine dimension types of the tensors
  std::vector< einsum_ir::dim_t > l_dim_types_left;
  std::vector< einsum_ir::dim_t > l_dim_types_right;
  std::vector< einsum_ir::dim_t > l_dim_types_out;
  BinaryContraction::dim_types( i_num_dims_left,
                                i_num_dims_right,
                                i_num_dims_out,
                                i_dim_ids_left,
                                i_dim_ids_right,
                                i_dim_ids_out,
                                &l_dim_types_left,
                                &l_dim_types_right,
                                &l_dim_types_out );

  int64_t l_di_left  = i_num_dims_left  - 1;
  int64_t l_di_right = i_num_dims_right - 1;
  int64_t l_di_out   = i_num_dims_out   - 1;

  // stride for contiguous storage
  int64_t l_stride_cont_left  = 1;
  int64_t l_stride_cont_right = 1;
  int64_t l_stride_cont_out   = 1;

  // determine C blocking
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      int64_t l_dim_id_left  = i_dim_ids_left[  l_di_left  ];
      int64_t l_dim_id_right = i_dim_ids_right[ l_di_right ];
      int64_t l_dim_id_out   = i_dim_ids_out[   l_di_out   ];

      int64_t l_dim_size = i_dim_sizes->at( l_dim_id_out );

      // determine if strides indicate contiguous storage
      int64_t l_stride_left  = i_strides_left->at(  l_dim_id_left  );
      int64_t l_stride_right = i_strides_right->at( l_dim_id_right );
      int64_t l_stride_out   = i_strides_out->at(   l_dim_id_out   );

      bool l_cont =    l_stride_left  == l_stride_cont_left
                    && l_stride_right == l_stride_cont_right
                    && l_stride_out   == l_stride_cont_out;

      if( !l_cont ) {
        return err_t::SUCCESS;
      }

      // block dimensions
      if(    l_dim_id_out == l_dim_id_left
          && l_dim_id_out == l_dim_id_right ) {
        o_dim_ids_cb->push_back( l_dim_id_out );

        l_size_cb *= l_dim_size;
        l_stride_cont_left  *= l_dim_size;
        l_stride_cont_right *= l_dim_size;
        l_stride_cont_out   *= l_dim_size;

        l_di_out--;
        l_di_left--;
        l_di_right--;
      }
      else {
        return err_t::SUCCESS;
      }
    }
    else {
      break;
    }
  }

  // determine M blocking
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      int64_t l_dim_id_left = i_dim_ids_left[ l_di_left ];
      int64_t l_dim_id_out  = i_dim_ids_out[  l_di_out ];

      int64_t l_dim_size = i_dim_sizes->at( l_dim_id_out );

      // determine if strides indicate contiguous storage
      int64_t l_stride_left = i_strides_left->at( l_dim_id_left );
      int64_t l_stride_out  = i_strides_out->at(  l_dim_id_out  );

      bool l_cont =    l_stride_left  == l_stride_cont_left
                    && l_stride_out   == l_stride_cont_out;

      if( !l_cont ) {
        break;
      }

      // block dimensions
      if(    l_dim_id_out == l_dim_id_left
          && (    l_size_mb              <  i_size_mb_min
               || l_size_mb * l_dim_size <= i_size_mb_max ) ) {
        o_dim_ids_mb->push_back( l_dim_id_out );
        l_size_mb *= l_dim_size;
        l_stride_cont_left  *= l_dim_size;
        l_stride_cont_out   *= l_dim_size;

        l_di_out--;
        l_di_left--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // seek the first K dimension after M in left tensor
  while( l_di_left >= 0 ) {
    int64_t l_dim_size = i_dim_sizes->at( i_dim_ids_left[ l_di_left ] );

    if( l_dim_types_left[l_di_left] == einsum_ir::dim_t::K ) {
      break;
    }
    else {
      l_stride_cont_left *= l_dim_size;
      l_di_left--;
    }
  }

  // reset stride for left tensor
  if( l_di_left >= 0 ) {
    l_stride_cont_left = i_strides_left->at( i_dim_ids_left[ l_di_left ] );
  }

  // determine K blocking
  while( l_di_left >= 0 ) {
    if( l_dim_types_left[l_di_left] == einsum_ir::dim_t::K ) {
      int64_t l_dim_id_left  = i_dim_ids_left[ l_di_left ];
      int64_t l_dim_id_right = i_dim_ids_right[ l_di_right ];

      int64_t l_dim_size = i_dim_sizes->at( l_dim_id_left );

      // determine if strides indicate contiguous storage
      int64_t l_stride_left  = i_strides_left->at(  l_dim_id_left  );
      int64_t l_stride_right = i_strides_right->at( l_dim_id_right );

      bool l_cont =    l_stride_left  == l_stride_cont_left
                    && l_stride_right == l_stride_cont_right;

      if( !l_cont ) {
        break;
      }

      // block dimensions
      if(    l_dim_id_left == l_dim_id_right
          && (    l_size_kb              <  i_size_kb_min
               || l_size_kb * l_dim_size <= i_size_kb_max ) ) {
        o_dim_ids_kb->push_back( l_dim_id_left );
        l_size_kb *= l_dim_size;
        l_stride_cont_left  *= l_dim_size;
        l_stride_cont_right *= l_dim_size;

        l_di_left--;
        l_di_right--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // seek first N dimension after M in output tensor
  while( l_di_out >= 0 ) {
    int64_t l_dim_size = i_dim_sizes->at( i_dim_ids_out[ l_di_out ] );

    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      break;
    }
    else {
      l_stride_cont_out *= l_dim_size;
      l_di_out--;
    }
  }

  // seek first N dimension after K in right tensor
  while( l_di_right >= 0 ) {
    int64_t l_dim_size = i_dim_sizes->at( i_dim_ids_right[ l_di_right ] );

    if( l_dim_types_right[l_di_right] == einsum_ir::dim_t::N ) {
      break;
    }
    else {
      l_stride_cont_right *= l_dim_size;
      l_di_right--;
    }
  }

  // reset stride for right and output tensors
  if( l_di_right >= 0 ) {
    l_stride_cont_right = i_strides_right->at( i_dim_ids_right[ l_di_right ] );
  }
  if( l_di_out >= 0 ) {
    l_stride_cont_out = i_strides_out->at( i_dim_ids_out[ l_di_out ] );
  }

  // determine N blocking
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      int64_t l_dim_id_out   = i_dim_ids_out[  l_di_out ];
      int64_t l_dim_id_right = i_dim_ids_right[ l_di_right ];

      int64_t l_dim_size = i_dim_sizes->at( l_dim_id_out );

      // determine if strides indicate contiguous storage
      int64_t l_stride_right = i_strides_right->at( l_dim_id_right );
      int64_t l_stride_out   = i_strides_out->at(   l_dim_id_out   );

      bool l_cont =    l_stride_right == l_stride_cont_right
                    && l_stride_out   == l_stride_cont_out;

      if( !l_cont ) {
        break;
      }

      // block dimensions
      if(    l_dim_id_out == l_dim_id_right
          && (    l_size_nb              <  i_size_nb_min
               || l_size_nb * l_dim_size <= i_size_nb_max ) ) {
        o_dim_ids_nb->push_back( l_dim_id_out );
        l_size_nb *= l_dim_size;
        l_stride_cont_right *= l_dim_size;
        l_stride_cont_out   *= l_dim_size;

        l_di_out--;
        l_di_right--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // transpose dimensions to match row-major
  std::reverse( o_dim_ids_cb->begin(), o_dim_ids_cb->end() );
  std::reverse( o_dim_ids_mb->begin(), o_dim_ids_mb->end() );
  std::reverse( o_dim_ids_nb->begin(), o_dim_ids_nb->end() );
  std::reverse( o_dim_ids_kb->begin(), o_dim_ids_kb->end() );

  return err_t::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::blocking_left_x_cb_kb_mb_right_x_cb_nb_kb_out_nb_x_mb_cb( int64_t                              i_size_mb_min,
                                                                                                                 int64_t                              i_size_mb_max,
                                                                                                                 int64_t                              i_size_nb_min,
                                                                                                                 int64_t                              i_size_nb_max,
                                                                                                                 int64_t                              i_size_kb_min,
                                                                                                                 int64_t                              i_size_kb_max,
                                                                                                                 int64_t                              i_num_dims_left,
                                                                                                                 int64_t                              i_num_dims_right,
                                                                                                                 int64_t                              i_num_dims_out,
                                                                                                                 int64_t                      const * i_dim_ids_left,
                                                                                                                 int64_t                      const * i_dim_ids_right,
                                                                                                                 int64_t                      const * i_dim_ids_out,
                                                                                                                 std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                                                                 std::map< int64_t, int64_t > const * i_strides_left,
                                                                                                                 std::map< int64_t, int64_t > const * i_strides_right,
                                                                                                                 std::map< int64_t, int64_t > const * i_strides_out,
                                                                                                                 std::vector< int64_t >             * o_dim_ids_cb,
                                                                                                                 std::vector< int64_t >             * o_dim_ids_mb,
                                                                                                                 std::vector< int64_t >             * o_dim_ids_nb,
                                                                                                                 std::vector< int64_t >             * o_dim_ids_kb ) {
  int64_t l_size_cb = 1;
  int64_t l_size_mb = 1;
  int64_t l_size_nb = 1;
  int64_t l_size_kb = 1;

  // init output
  o_dim_ids_cb->resize( 0 );
  o_dim_ids_mb->resize( 0 );
  o_dim_ids_nb->resize( 0 );
  o_dim_ids_kb->resize( 0 );

  // determine dimension types of the tensors
  std::vector< einsum_ir::dim_t > l_dim_types_left;
  std::vector< einsum_ir::dim_t > l_dim_types_right;
  std::vector< einsum_ir::dim_t > l_dim_types_out;
  BinaryContraction::dim_types( i_num_dims_left,
                                i_num_dims_right,
                                i_num_dims_out,
                                i_dim_ids_left,
                                i_dim_ids_right,
                                i_dim_ids_out,
                                &l_dim_types_left,
                                &l_dim_types_right,
                                &l_dim_types_out );

  int64_t l_di_left  = i_num_dims_left  - 1;
  int64_t l_di_right = i_num_dims_right - 1;
  int64_t l_di_out   = i_num_dims_out   - 1;

  // stride for contiguous storage
  int64_t l_stride_cont_left  = 1;
  int64_t l_stride_cont_right = 1;
  int64_t l_stride_cont_out   = 1;

  // jump over M dimensions in left tensor
  while( l_di_left >= 0 ) {
    if( l_dim_types_left[l_di_left] != einsum_ir::dim_t::M ) {
      break;
    }
    else {
      l_stride_cont_left *= i_dim_sizes->at( i_dim_ids_left[ l_di_left ] );
      l_di_left--;
    }
  }

  // jump over K dimensions in left tensor
  while( l_di_left >= 0 ) {
    if( l_dim_types_left[l_di_left] != einsum_ir::dim_t::K ) {
      break;
    }
    else {
      l_stride_cont_left *= i_dim_sizes->at( i_dim_ids_left[ l_di_left ] );
      l_di_left--;
    }
  }

  // jump over K dimensions in right tensor
  while( l_di_right >= 0 ) {
    if( l_dim_types_right[l_di_right] != einsum_ir::dim_t::K ) {
      break;
    }
    else {
      l_stride_cont_right *= i_dim_sizes->at( i_dim_ids_right[ l_di_right ] );
      l_di_right--;
    }
  }

  // jump over N dimensions in right tensor
  while( l_di_right >= 0 ) {
    if( l_dim_types_right[l_di_right] != einsum_ir::dim_t::N ) {
      break;
    }
    else {
      l_stride_cont_right *= i_dim_sizes->at( i_dim_ids_right[ l_di_right ] );
      l_di_right--;
    }
  }

  // determine C blocking
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      int64_t l_dim_id_left  = i_dim_ids_left[  l_di_left  ];
      int64_t l_dim_id_right = i_dim_ids_right[ l_di_right ];
      int64_t l_dim_id_out   = i_dim_ids_out[   l_di_out   ];

      int64_t l_dim_size = i_dim_sizes->at( l_dim_id_out );

      // determine if strides indicate contiguous storage
      int64_t l_stride_left  = i_strides_left->at(  l_dim_id_left  );
      int64_t l_stride_right = i_strides_right->at( l_dim_id_right );
      int64_t l_stride_out   = i_strides_out->at(   l_dim_id_out   );

      bool l_cont =    l_stride_left  == l_stride_cont_left
                    && l_stride_right == l_stride_cont_right
                    && l_stride_out   == l_stride_cont_out;

      if( !l_cont ) {
        return err_t::SUCCESS;
      }

      // block dimensions
      if(    l_dim_id_out == l_dim_id_left
          && l_dim_id_out == l_dim_id_right ) {
        o_dim_ids_cb->push_back( l_dim_id_out );

        l_size_cb *= l_dim_size;
        l_stride_cont_left  *= l_dim_size;
        l_stride_cont_right *= l_dim_size;
        l_stride_cont_out   *= l_dim_size;

        l_di_out--;
        l_di_left--;
        l_di_right--;
      }
      else {
        return err_t::SUCCESS;
      }
    }
    else {
      break;
    }
  }

  // reset position and stride for left tensor
  l_di_left = i_num_dims_left - 1;
  l_stride_cont_left = 1;

  // reset posotion and stride for right tensor
  l_di_right = i_num_dims_right - 1;
  l_stride_cont_right = 1;

  // determine M blocking
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      int64_t l_dim_id_left = i_dim_ids_left[ l_di_left ];
      int64_t l_dim_id_out  = i_dim_ids_out[  l_di_out  ];

      int64_t l_dim_size = i_dim_sizes->at( l_dim_id_out );

      // determine if strides indicate contiguous storage
      int64_t l_stride_left = i_strides_left->at( l_dim_id_left );
      int64_t l_stride_out  = i_strides_out->at(  l_dim_id_out  );

      bool l_cont =    l_stride_left == l_stride_cont_left
                    && l_stride_out  == l_stride_cont_out;

      if( !l_cont ) {
        break;
      }

      // block dimensions
      if(    l_dim_id_out == l_dim_id_left
          && (    l_size_mb              <  i_size_mb_min
               || l_size_mb * l_dim_size <= i_size_mb_max ) ) {
        o_dim_ids_mb->push_back( l_dim_id_out );

        l_size_mb *= l_dim_size;
        l_stride_cont_left *= l_dim_size;
        l_stride_cont_out  *= l_dim_size;

        l_di_out--;
        l_di_left--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // determine K blocking
  while( l_di_left >= 0 ) {
    if( l_dim_types_left[l_di_left] == einsum_ir::dim_t::K ) {
      int64_t l_dim_id_left  = i_dim_ids_left[ l_di_left ];
      int64_t l_dim_id_right = i_dim_ids_right[ l_di_right ];

      int64_t l_dim_size = i_dim_sizes->at( l_dim_id_left );

      // determine if strides indicate contiguous storage
      int64_t l_stride_left  = i_strides_left->at(  l_dim_id_left  );
      int64_t l_stride_right = i_strides_right->at( l_dim_id_right );

      bool l_cont =    l_stride_left  == l_stride_cont_left
                    && l_stride_right == l_stride_cont_right;

      if( !l_cont ) {
        break;
      }

      // block dimensions
      if(    l_dim_id_left == l_dim_id_right
          && (    l_size_kb              <  i_size_kb_min
               || l_size_kb * l_dim_size <= i_size_kb_max ) ) {
        o_dim_ids_kb->push_back( l_dim_id_left );

        l_size_kb *= l_dim_size;
        l_stride_cont_left  *= l_dim_size;
        l_stride_cont_right *= l_dim_size;

        l_di_left--;
        l_di_right--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // seek first N dimension after M in output tensor
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      break;
    }
    else {
      l_di_out--;
    }
  }

  // reset stride for output tensor
  if( l_di_out >= 0 ) {
    l_stride_cont_out = i_strides_out->at( i_dim_ids_out[ l_di_out ] );
  }

  // determine N blocking
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      int64_t l_dim_id_out   = i_dim_ids_out[  l_di_out ];
      int64_t l_dim_id_right = i_dim_ids_right[ l_di_right ];

      int64_t l_dim_size = i_dim_sizes->at( l_dim_id_out );

      // determine if strides indicate contiguous storage
      int64_t l_stride_right = i_strides_right->at( l_dim_id_right );
      int64_t l_stride_out   = i_strides_out->at(   l_dim_id_out   );

      bool l_cont =    l_stride_right == l_stride_cont_right
                    && l_stride_out   == l_stride_cont_out;

      if( !l_cont ) {
        break;
      }

      // block dimensions
      if(    l_dim_id_out == l_dim_id_right
          && (    l_size_nb              <  i_size_nb_min
               || l_size_nb * l_dim_size <= i_size_nb_max ) ) {
        o_dim_ids_nb->push_back( l_dim_id_out );

        l_size_nb *= l_dim_size;
        l_stride_cont_right *= l_dim_size;
        l_stride_cont_out   *= l_dim_size;

        l_di_out--;
        l_di_right--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // transpose dimensions to match row-major
  std::reverse( o_dim_ids_cb->begin(), o_dim_ids_cb->end() );
  std::reverse( o_dim_ids_mb->begin(), o_dim_ids_mb->end() );
  std::reverse( o_dim_ids_nb->begin(), o_dim_ids_nb->end() );
  std::reverse( o_dim_ids_kb->begin(), o_dim_ids_kb->end() );

  return err_t::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::blocking( primblo_t                            i_primitive_blocking,
                                                                 int64_t                              i_num_dims_left,
                                                                 int64_t                              i_num_dims_right,
                                                                 int64_t                              i_num_dims_out,
                                                                 int64_t                      const * i_dim_ids_left,
                                                                 int64_t                      const * i_dim_ids_right,
                                                                 int64_t                      const * i_dim_ids_out,
                                                                 std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                 std::map< int64_t, int64_t > const * i_strides_left,
                                                                 std::map< int64_t, int64_t > const * i_strides_right,
                                                                 std::map< int64_t, int64_t > const * i_strides_out,
                                                                 std::vector< int64_t >             * o_dim_ids_cb,
                                                                 std::vector< int64_t >             * o_dim_ids_mb,
                                                                 std::vector< int64_t >             * o_dim_ids_nb,
                                                                 std::vector< int64_t >             * o_dim_ids_kb ) {
  err_t l_err = err_t::TENSOR_BLOCKING_FAILED;

  std::map< int64_t, int64_t > l_strides_left;
  std::map< int64_t, int64_t > l_strides_right;
  std::map< int64_t, int64_t > l_strides_out;

  // assume contiguous storage and determine strides if not provided
  if( i_strides_left == nullptr ) {
    int64_t l_stride = 1;

    for( int64_t l_di = i_num_dims_left - 1; l_di >= 0; l_di-- ) {
      int64_t l_id_left = i_dim_ids_left[l_di];
      l_strides_left[l_id_left] = l_stride;
      l_stride *= i_dim_sizes->at( l_id_left );
    }
  }

  if( i_strides_right == nullptr ) {
    int64_t l_stride = 1;

    for( int64_t l_di = i_num_dims_right - 1; l_di >= 0; l_di-- ) {
      int64_t l_id_right = i_dim_ids_right[l_di];
      l_strides_right[l_id_right] = l_stride;
      l_stride *= i_dim_sizes->at( l_id_right );
    }
  }

  if( i_strides_out == nullptr ) {
    int64_t l_stride = 1;

    for( int64_t l_di = i_num_dims_out - 1; l_di >= 0; l_di-- ) {
      int64_t l_id_out = i_dim_ids_out[l_di];
      l_strides_out[l_id_out] = l_stride;
      l_stride *= i_dim_sizes->at( l_id_out );
    }
  }

  std::map< int64_t, int64_t > const * l_strides_ptr_left  = (i_strides_left  != nullptr) ? i_strides_left  : &l_strides_left;
  std::map< int64_t, int64_t > const * l_strides_ptr_right = (i_strides_right != nullptr) ? i_strides_right : &l_strides_right;
  std::map< int64_t, int64_t > const * l_strides_ptr_out   = (i_strides_out   != nullptr) ? i_strides_out   : &l_strides_out;

  // derive blocking
  if( i_primitive_blocking == primblo_t::LEFT_KB_X_MB_CB_RIGHT_NB_X_KB_CB_OUT_NB_X_MB_CB ) {
    l_err = blocking_left_kb_x_mb_cb_right_nb_x_kb_cb_out_nb_x_mb_cb( m_size_mb_min,
                                                                      m_size_mb_max,
                                                                      m_size_nb_min,
                                                                      m_size_nb_max,
                                                                      m_size_kb_min,
                                                                      m_size_kb_max,
                                                                      i_num_dims_left,
                                                                      i_num_dims_right,
                                                                      i_num_dims_out,
                                                                      i_dim_ids_left,
                                                                      i_dim_ids_right,
                                                                      i_dim_ids_out,
                                                                      i_dim_sizes,
                                                                      l_strides_ptr_left,
                                                                      l_strides_ptr_right,
                                                                      l_strides_ptr_out,
                                                                      o_dim_ids_cb,
                                                                      o_dim_ids_mb,
                                                                      o_dim_ids_nb,
                                                                      o_dim_ids_kb );
  }
  else if( i_primitive_blocking == primblo_t::LEFT_X_CB_KB_MB_RIGHT_X_CB_NB_KB_OUT_NB_X_MB_CB ) {
    l_err = blocking_left_x_cb_kb_mb_right_x_cb_nb_kb_out_nb_x_mb_cb( m_size_mb_min,
                                                                      m_size_mb_max,
                                                                      m_size_nb_min,
                                                                      m_size_nb_max,
                                                                      m_size_kb_min,
                                                                      m_size_kb_max,
                                                                      i_num_dims_left,
                                                                      i_num_dims_right,
                                                                      i_num_dims_out,
                                                                      i_dim_ids_left,
                                                                      i_dim_ids_right,
                                                                      i_dim_ids_out,
                                                                      i_dim_sizes,
                                                                      l_strides_ptr_left,
                                                                      l_strides_ptr_right,
                                                                      l_strides_ptr_out,
                                                                      o_dim_ids_cb,
                                                                      o_dim_ids_mb,
                                                                      o_dim_ids_nb,
                                                                      o_dim_ids_kb );
  }

  return l_err;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::reorder_left_bc_bm_bk_bi_kb_mb_cb_right_bc_bn_bk_bj_nb_kb_cb_out_native( int64_t                              i_size_cb_min,
                                                                                                                                int64_t                              i_size_cb_max,
                                                                                                                                int64_t                              i_size_mb_min,
                                                                                                                                int64_t                              i_size_mb_max,
                                                                                                                                int64_t                              i_size_nb_min,
                                                                                                                                int64_t                              i_size_nb_max,
                                                                                                                                int64_t                              i_size_kb_min,
                                                                                                                                int64_t                              i_size_kb_max,
                                                                                                                                int64_t                              i_num_dims_left,
                                                                                                                                int64_t                              i_num_dims_right,
                                                                                                                                int64_t                              i_num_dims_out,
                                                                                                                                std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                                                                                int64_t                            * io_dim_ids_left,
                                                                                                                                int64_t                            * io_dim_ids_right,
                                                                                                                                int64_t                            * io_dim_ids_out ) {
  // determine dimension types of the tensors
  std::vector< einsum_ir::dim_t > l_dim_types_left;
  std::vector< einsum_ir::dim_t > l_dim_types_right;
  std::vector< einsum_ir::dim_t > l_dim_types_out;
  BinaryContraction::dim_types( i_num_dims_left,
                                i_num_dims_right,
                                i_num_dims_out,
                                io_dim_ids_left,
                                io_dim_ids_right,
                                io_dim_ids_out,
                                &l_dim_types_left,
                                &l_dim_types_right,
                                &l_dim_types_out );

  // blocked dimension ids
  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  // non-blocked dimension ids
  std::vector< int64_t > l_dim_ids_bc;
  std::vector< int64_t > l_dim_ids_bm;
  std::vector< int64_t > l_dim_ids_bn;
  std::vector< int64_t > l_dim_ids_bk;
  std::vector< int64_t > l_dim_ids_bi;
  std::vector< int64_t > l_dim_ids_bj;

  // derive blocked C dimensions
  int64_t l_di_out = i_num_dims_out - 1;
  int64_t l_block_dim_size = 1;
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      int64_t l_dim_id   = io_dim_ids_out[ l_di_out ];
      int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

      if(    l_block_dim_size < i_size_cb_min
          || l_block_dim_size * l_dim_size <= i_size_cb_max ) {
        l_block_dim_size *= l_dim_size;
        l_dim_ids_cb.push_back( l_dim_id );
        l_di_out--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // derive blocked M dimensions
  l_block_dim_size = 1;
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      int64_t l_dim_id   = io_dim_ids_out[ l_di_out ];
      int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

      if(    l_block_dim_size < i_size_mb_min
          || l_block_dim_size * l_dim_size <= i_size_mb_max ) {
        l_block_dim_size *= l_dim_size;
        l_dim_ids_mb.push_back( l_dim_id );
        l_di_out--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // seek the first N dimension after M
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      l_dim_ids_bc.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      l_dim_ids_bm.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      break;
    }
    else {
      return einsum_ir::err_t::DIMENSION_ORDERING_FAILED;
    }
  }

  // derive blocked N dimensions
  l_block_dim_size = 1;
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      int64_t l_dim_id   = io_dim_ids_out[ l_di_out ];
      int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

      if(    l_block_dim_size < i_size_nb_min
          || l_block_dim_size * l_dim_size <= i_size_nb_max ) {
        l_block_dim_size *= l_dim_size;
        l_dim_ids_nb.push_back( l_dim_id );
        l_di_out--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // assign remaining non-blocked dimensions of the output tensor
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      l_dim_ids_bc.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      l_dim_ids_bm.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      l_dim_ids_bn.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else {
      return einsum_ir::err_t::DIMENSION_ORDERING_FAILED;
    }
  }

  // extract K dimension ids
  std::vector< int64_t > l_dim_ids_k;
  for( int64_t l_id = 0; l_id < i_num_dims_left; l_id++ ) {
    if( l_dim_types_left[l_id] == einsum_ir::dim_t::K ) {
      l_dim_ids_k.push_back( io_dim_ids_left[l_id] );
    }
  }

  // sort K dimension ids by their size
  std::sort( l_dim_ids_k.begin(),
             l_dim_ids_k.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) > i_dim_sizes->at( i_rhs );
             } );

  // derive blocked K dimensions
  l_block_dim_size = 1;
  for( std::size_t l_id = 0; l_id < l_dim_ids_k.size(); l_id++ ) {
    int64_t l_dim_size = i_dim_sizes->at( l_dim_ids_k[l_id] );

    if(    l_block_dim_size < i_size_kb_min
        || l_block_dim_size * l_dim_size <= i_size_kb_max ) {
      l_block_dim_size *= l_dim_size;
      l_dim_ids_kb.push_back( l_dim_ids_k[l_id] );
    }
    else {
      break;
    }
  }
  std::reverse( l_dim_ids_kb.begin(), l_dim_ids_kb.end() );

  // derive non-blocked K dimensions
  l_dim_ids_bk = std::vector< int64_t >( l_dim_ids_k.begin() + l_dim_ids_kb.size(),
                                         l_dim_ids_k.end() );

  // extract I dimension ids
  for( int64_t l_id = 0; l_id < i_num_dims_left; l_id++ ) {
    if( l_dim_types_left[l_id] == einsum_ir::dim_t::I ) {
      l_dim_ids_bi.push_back( io_dim_ids_left[l_id] );
    }
  }

  // sort I dimension ids by their size
  std::sort( l_dim_ids_bi.begin(),
             l_dim_ids_bi.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) <= i_dim_sizes->at( i_rhs );
             } );

  // extract J dimension ids
  for( int64_t l_id = 0; l_id < i_num_dims_right; l_id++ ) {
    if( l_dim_types_right[l_id] == einsum_ir::dim_t::J ) {
      l_dim_ids_bj.push_back( io_dim_ids_right[l_id] );
    }
  }

  // sort J dimension ids by their size
  std::sort( l_dim_ids_bj.begin(),
             l_dim_ids_bj.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) <= i_dim_sizes->at( i_rhs );
             } );

  // perform the reordering of the input tensors' dimensions
  int64_t l_di_left  = i_num_dims_left  - 1;
  int64_t l_di_right = i_num_dims_right - 1;

  // assign blocked C dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_cb.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_cb[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_cb[l_di];
    l_di_left--;
    l_di_right--;
  }

  // assign blocked M dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_mb.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_mb[l_di];
    l_di_left--;
  }

  // assign blocked K dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_kb.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_kb[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_kb[l_di];
    l_di_left--;
    l_di_right--;
  }

  // assign blocked N dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_nb.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_nb[l_di];
    l_di_right--;
  }

  // assign non-blocked I dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bi.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_bi[l_di];
    l_di_left--;
  }

  // assign non-blocked J dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bj.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_bj[l_di];
    l_di_right--;
  }

  // asssign non-blocked K dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bk.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_bk[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_bk[l_di];
    l_di_left--;
    l_di_right--;
  }

  // assign non-blocked M dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bm.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_bm[l_di];
    l_di_left--;
  }

  // assign non-blocked N dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bn.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_bn[l_di];
    l_di_right--;
  }

  // assign non-blocked C dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bc.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_bc[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_bc[l_di];
    l_di_left--;
    l_di_right--;
  }

  return err_t::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::reorder_left_bc_bm_bk_bi_kb_mb_right_bc_bn_bk_bj_nb_kb_out_native( int64_t                              i_size_mb_min,
                                                                                                                          int64_t                              i_size_mb_max,
                                                                                                                          int64_t                              i_size_nb_min,
                                                                                                                          int64_t                              i_size_nb_max,
                                                                                                                          int64_t                              i_size_kb_min,
                                                                                                                          int64_t                              i_size_kb_max,
                                                                                                                          int64_t                              i_num_dims_left,
                                                                                                                          int64_t                              i_num_dims_right,
                                                                                                                          int64_t                              i_num_dims_out,
                                                                                                                          std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                                                                          int64_t                            * io_dim_ids_left,
                                                                                                                          int64_t                            * io_dim_ids_right,
                                                                                                                          int64_t                            * io_dim_ids_out ) {
  // determine dimension types of the tensors
  std::vector< einsum_ir::dim_t > l_dim_types_left;
  std::vector< einsum_ir::dim_t > l_dim_types_right;
  std::vector< einsum_ir::dim_t > l_dim_types_out;
  BinaryContraction::dim_types( i_num_dims_left,
                                i_num_dims_right,
                                i_num_dims_out,
                                io_dim_ids_left,
                                io_dim_ids_right,
                                io_dim_ids_out,
                                &l_dim_types_left,
                                &l_dim_types_right,
                                &l_dim_types_out );

  // blocked dimension ids
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  // non-blocked dimension ids
  std::vector< int64_t > l_dim_ids_bc;
  std::vector< int64_t > l_dim_ids_bm;
  std::vector< int64_t > l_dim_ids_bn;
  std::vector< int64_t > l_dim_ids_bk;
  std::vector< int64_t > l_dim_ids_bi;
  std::vector< int64_t > l_dim_ids_bj;

  // derive blocked M dimensions
  int64_t l_di_out = i_num_dims_out - 1;
  int64_t l_block_dim_size = 1;
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      int64_t l_dim_id   = io_dim_ids_out[ l_di_out ];
      int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

      if(    l_block_dim_size < i_size_mb_min
          || l_block_dim_size * l_dim_size <= i_size_mb_max ) {
        l_block_dim_size *= l_dim_size;
        l_dim_ids_mb.push_back( l_dim_id );
        l_di_out--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // seek the first N dimension after M
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      l_dim_ids_bc.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      l_dim_ids_bm.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      break;
    }
    else {
      return einsum_ir::err_t::DIMENSION_ORDERING_FAILED;
    }
  }

  // derive blocked N dimensions
  l_block_dim_size = 1;
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      int64_t l_dim_id   = io_dim_ids_out[ l_di_out ];
      int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

      if(    l_block_dim_size < i_size_nb_min
          || l_block_dim_size * l_dim_size <= i_size_nb_max ) {
        l_block_dim_size *= l_dim_size;
        l_dim_ids_nb.push_back( l_dim_id );
        l_di_out--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // assign remaining non-blocked dimensions of the output tensor
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      l_dim_ids_bc.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      l_dim_ids_bm.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      l_dim_ids_bn.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else {
      return einsum_ir::err_t::DIMENSION_ORDERING_FAILED;
    }
  }

  // extract K dimension ids
  std::vector< int64_t > l_dim_ids_k;
  for( int64_t l_id = 0; l_id < i_num_dims_left; l_id++ ) {
    if( l_dim_types_left[l_id] == einsum_ir::dim_t::K ) {
      l_dim_ids_k.push_back( io_dim_ids_left[l_id] );
    }
  }

  // sort K dimension ids by their size
  std::sort( l_dim_ids_k.begin(),
             l_dim_ids_k.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) > i_dim_sizes->at( i_rhs );
             } );

  // derive blocked K dimensions
  l_block_dim_size = 1;
  for( std::size_t l_id = 0; l_id < l_dim_ids_k.size(); l_id++ ) {
    int64_t l_dim_size = i_dim_sizes->at( l_dim_ids_k[l_id] );

    if(    l_block_dim_size < i_size_kb_min
        || l_block_dim_size * l_dim_size <= i_size_kb_max ) {
      l_block_dim_size *= l_dim_size;
      l_dim_ids_kb.push_back( l_dim_ids_k[l_id] );
    }
    else {
      break;
    }
  }
  std::reverse( l_dim_ids_kb.begin(), l_dim_ids_kb.end() );

  // derive non-blocked K dimensions
  l_dim_ids_bk = std::vector< int64_t >( l_dim_ids_k.begin() + l_dim_ids_kb.size(),
                                         l_dim_ids_k.end() );

  // extract I dimension ids
  for( int64_t l_id = 0; l_id < i_num_dims_left; l_id++ ) {
    if( l_dim_types_left[l_id] == einsum_ir::dim_t::I ) {
      l_dim_ids_bi.push_back( io_dim_ids_left[l_id] );
    }
  }

  // sort I dimension ids by their size
  std::sort( l_dim_ids_bi.begin(),
             l_dim_ids_bi.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) <= i_dim_sizes->at( i_rhs );
             } );

  // extract J dimension ids
  for( int64_t l_id = 0; l_id < i_num_dims_right; l_id++ ) {
    if( l_dim_types_right[l_id] == einsum_ir::dim_t::J ) {
      l_dim_ids_bj.push_back( io_dim_ids_right[l_id] );
    }
  }

  // sort J dimension ids by their size
  std::sort( l_dim_ids_bj.begin(),
             l_dim_ids_bj.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) <= i_dim_sizes->at( i_rhs );
             } );

  // perform the reordering of the input tensors' dimensions
  int64_t l_di_left  = i_num_dims_left  - 1;
  int64_t l_di_right = i_num_dims_right - 1;

  // assign blocked M dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_mb.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_mb[l_di];
    l_di_left--;
  }

  // assign blocked K dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_kb.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_kb[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_kb[l_di];
    l_di_left--;
    l_di_right--;
  }

  // assign blocked N dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_nb.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_nb[l_di];
    l_di_right--;
  }

  // assign non-blocked I dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bi.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_bi[l_di];
    l_di_left--;
  }

  // assign non-blocked J dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bj.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_bj[l_di];
    l_di_right--;
  }

  // asssign non-blocked K dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bk.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_bk[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_bk[l_di];
    l_di_left--;
    l_di_right--;
  }

  // assign non-blocked M dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bm.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_bm[l_di];
    l_di_left--;
  }

  // assign non-blocked N dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bn.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_bn[l_di];
    l_di_right--;
  }

  // assign non-blocked C dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bc.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_bc[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_bc[l_di];
    l_di_left--;
    l_di_right--;
  }

  return err_t::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::reorder_left_bc_bm_bk_bi_cb_kb_mb_right_bc_bn_bk_bj_cb_nb_kb_out_native( int64_t                              i_size_mb_min,
                                                                                                                                int64_t                              i_size_mb_max,
                                                                                                                                int64_t                              i_size_nb_min,
                                                                                                                                int64_t                              i_size_nb_max,
                                                                                                                                int64_t                              i_size_kb_min,
                                                                                                                                int64_t                              i_size_kb_max,
                                                                                                                                int64_t                              i_num_dims_left,
                                                                                                                                int64_t                              i_num_dims_right,
                                                                                                                                int64_t                              i_num_dims_out,
                                                                                                                                std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                                                                                int64_t                            * io_dim_ids_left,
                                                                                                                                int64_t                            * io_dim_ids_right,
                                                                                                                                int64_t                            * io_dim_ids_out ) {
  // determine dimension types of the tensors
  std::vector< einsum_ir::dim_t > l_dim_types_left;
  std::vector< einsum_ir::dim_t > l_dim_types_right;
  std::vector< einsum_ir::dim_t > l_dim_types_out;
  BinaryContraction::dim_types( i_num_dims_left,
                                i_num_dims_right,
                                i_num_dims_out,
                                io_dim_ids_left,
                                io_dim_ids_right,
                                io_dim_ids_out,
                                &l_dim_types_left,
                                &l_dim_types_right,
                                &l_dim_types_out );

  // blocked dimension ids
  std::vector< int64_t > l_dim_ids_cb;
  std::vector< int64_t > l_dim_ids_mb;
  std::vector< int64_t > l_dim_ids_nb;
  std::vector< int64_t > l_dim_ids_kb;

  // non-blocked dimension ids
  std::vector< int64_t > l_dim_ids_bc;
  std::vector< int64_t > l_dim_ids_bm;
  std::vector< int64_t > l_dim_ids_bn;
  std::vector< int64_t > l_dim_ids_bk;
  std::vector< int64_t > l_dim_ids_bi;
  std::vector< int64_t > l_dim_ids_bj;

  // derive blocked C dimensions
  int64_t l_di_out = i_num_dims_out - 1;
  int64_t l_block_dim_size = 1;
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      int64_t l_dim_id   = io_dim_ids_out[ l_di_out ];
      int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

      l_block_dim_size *= l_dim_size;
      l_dim_ids_cb.push_back( l_dim_id );
      l_di_out--;
    }
    else {
      break;
    }
  }

  // derive blocked M dimensions
  l_block_dim_size = 1;
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      int64_t l_dim_id   = io_dim_ids_out[ l_di_out ];
      int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

      if(    l_block_dim_size < i_size_mb_min
          || l_block_dim_size * l_dim_size <= i_size_mb_max ) {
        l_block_dim_size *= l_dim_size;
        l_dim_ids_mb.push_back( l_dim_id );
        l_di_out--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // seek the first N dimension after M
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      l_dim_ids_bc.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      l_dim_ids_bm.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      break;
    }
    else {
      return einsum_ir::err_t::DIMENSION_ORDERING_FAILED;
    }
  }

  // derive blocked N dimensions
  l_block_dim_size = 1;
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      int64_t l_dim_id   = io_dim_ids_out[ l_di_out ];
      int64_t l_dim_size = i_dim_sizes->at( l_dim_id );

      if(    l_block_dim_size < i_size_nb_min
          || l_block_dim_size * l_dim_size <= i_size_nb_max ) {
        l_block_dim_size *= l_dim_size;
        l_dim_ids_nb.push_back( l_dim_id );
        l_di_out--;
      }
      else {
        break;
      }
    }
    else {
      break;
    }
  }

  // assign remaining non-blocked dimensions of the output tensor
  while( l_di_out >= 0 ) {
    if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::C ) {
      l_dim_ids_bc.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::M ) {
      l_dim_ids_bm.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else if( l_dim_types_out[l_di_out] == einsum_ir::dim_t::N ) {
      l_dim_ids_bn.push_back( io_dim_ids_out[l_di_out] );
      l_di_out--;
    }
    else {
      return einsum_ir::err_t::DIMENSION_ORDERING_FAILED;
    }
  }

  // extract K dimension ids
  std::vector< int64_t > l_dim_ids_k;
  for( int64_t l_id = 0; l_id < i_num_dims_left; l_id++ ) {
    if( l_dim_types_left[l_id] == einsum_ir::dim_t::K ) {
      l_dim_ids_k.push_back( io_dim_ids_left[l_id] );
    }
  }

  // sort K dimension ids by their size
  std::sort( l_dim_ids_k.begin(),
             l_dim_ids_k.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) > i_dim_sizes->at( i_rhs );
             } );

  // derive blocked K dimensions
  l_block_dim_size = 1;
  for( std::size_t l_id = 0; l_id < l_dim_ids_k.size(); l_id++ ) {
    int64_t l_dim_size = i_dim_sizes->at( l_dim_ids_k[l_id] );

    if(    l_block_dim_size < i_size_kb_min
        || l_block_dim_size * l_dim_size <= i_size_kb_max ) {
      l_block_dim_size *= l_dim_size;
      l_dim_ids_kb.push_back( l_dim_ids_k[l_id] );
    }
    else {
      break;
    }
  }
  std::reverse( l_dim_ids_kb.begin(), l_dim_ids_kb.end() );

  // derive non-blocked K dimensions
  l_dim_ids_bk = std::vector< int64_t >( l_dim_ids_k.begin() + l_dim_ids_kb.size(),
                                         l_dim_ids_k.end() );

  // extract I dimension ids
  for( int64_t l_id = 0; l_id < i_num_dims_left; l_id++ ) {
    if( l_dim_types_left[l_id] == einsum_ir::dim_t::I ) {
      l_dim_ids_bi.push_back( io_dim_ids_left[l_id] );
    }
  }

  // sort I dimension ids by their size
  std::sort( l_dim_ids_bi.begin(),
             l_dim_ids_bi.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) <= i_dim_sizes->at( i_rhs );
             } );

  // extract J dimension ids
  for( int64_t l_id = 0; l_id < i_num_dims_right; l_id++ ) {
    if( l_dim_types_right[l_id] == einsum_ir::dim_t::J ) {
      l_dim_ids_bj.push_back( io_dim_ids_right[l_id] );
    }
  }

  // sort J dimension ids by their size
  std::sort( l_dim_ids_bj.begin(),
             l_dim_ids_bj.end(),
             [i_dim_sizes]( int64_t i_lhs, int64_t i_rhs ) {
               return i_dim_sizes->at( i_lhs ) <= i_dim_sizes->at( i_rhs );
             } );

  // perform the reordering of the input tensors' dimensions
  int64_t l_di_left  = i_num_dims_left  - 1;
  int64_t l_di_right = i_num_dims_right - 1;

  // assign blocked M dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_mb.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_mb[l_di];
    l_di_left--;
  }

  // assign blocked K dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_kb.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_kb[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_kb[l_di];
    l_di_left--;
    l_di_right--;
  }

  // assign blocked N dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_nb.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_nb[l_di];
    l_di_right--;
  }

  // assign blocked C dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_cb.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_cb[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_cb[l_di];
    l_di_left--;
    l_di_right--;
  }

  // assign non-blocked I dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bi.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_bi[l_di];
    l_di_left--;
  }

  // assign non-blocked J dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bj.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_bj[l_di];
    l_di_right--;
  }

  // assign non-blocked K dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bk.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_bk[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_bk[l_di];
    l_di_left--;
    l_di_right--;
  }

  // assign non-blocked M dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bm.size(); l_di++ ) {
    io_dim_ids_left[l_di_left] = l_dim_ids_bm[l_di];
    l_di_left--;
  }

  // assign non-blocked N dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bn.size(); l_di++ ) {
    io_dim_ids_right[l_di_right] = l_dim_ids_bn[l_di];
    l_di_right--;
  }

  // assign non-blocked C dimensions
  for( std::size_t l_di = 0; l_di < l_dim_ids_bc.size(); l_di++ ) {
    io_dim_ids_left[l_di_left]   = l_dim_ids_bc[l_di];
    io_dim_ids_right[l_di_right] = l_dim_ids_bc[l_di];
    l_di_left--;
    l_di_right--;
  }

  return err_t::SUCCESS;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::reorder( tenord_t                             i_primitive_ordering,
                                                                int64_t                              i_num_dims_left,
                                                                int64_t                              i_num_dims_right,
                                                                int64_t                              i_num_dims_out,
                                                                std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                int64_t                            * io_dim_ids_left,
                                                                int64_t                            * io_dim_ids_right,
                                                                int64_t                            * io_dim_ids_out ) const {
  err_t l_err = err_t::DIMENSION_ORDERING_FAILED;

  if( i_primitive_ordering == tenord_t::LEFT_NATIVE_RIGHT_NATIVE_OUT_NATIVE ) {
    l_err = err_t::SUCCESS;
  }
  // TODO: adjust enum to fit routine
  else if( i_primitive_ordering == tenord_t::LEFT_BC_BM_BK_BI_KB_MB_RIGHT_BC_BN_BK_BJ_NB_KB_OUT_NATIVE ) {
    l_err = reorder_left_bc_bm_bk_bi_kb_mb_right_bc_bn_bk_bj_nb_kb_out_native( m_size_mb_min,
                                                                               m_size_mb_max,
                                                                               m_size_nb_min,
                                                                               m_size_nb_max,
                                                                               m_size_kb_min,
                                                                               m_size_kb_max,
                                                                               i_num_dims_left,
                                                                               i_num_dims_right,
                                                                               i_num_dims_out,
                                                                               i_dim_sizes,
                                                                               io_dim_ids_left,
                                                                               io_dim_ids_right,
                                                                               io_dim_ids_out );
  }
  // TODO: adjust enum to fit routine
  else if( i_primitive_ordering == tenord_t::LEFT_BC_BM_BK_BI_KB_MB_CB_RIGHT_BC_BN_BK_BJ_NB_KB_CB_OUT_NATIVE ) {
    l_err = reorder_left_bc_bm_bk_bi_kb_mb_cb_right_bc_bn_bk_bj_nb_kb_cb_out_native( m_size_cb_min,
                                                                                     m_size_cb_max,
                                                                                     m_size_mb_min,
                                                                                     m_size_mb_max,
                                                                                     m_size_nb_min,
                                                                                     m_size_nb_max,
                                                                                     m_size_kb_min,
                                                                                     m_size_kb_max,
                                                                                     i_num_dims_left,
                                                                                     i_num_dims_right,
                                                                                     i_num_dims_out,
                                                                                     i_dim_sizes,
                                                                                     io_dim_ids_left,
                                                                                     io_dim_ids_right,
                                                                                     io_dim_ids_out );
  }
  else if( i_primitive_ordering == tenord_t::LEFT_BC_BM_BK_BI_CB_KB_MB_RIGHT_BC_BN_BK_BJ_CB_NB_KB_OUT_NATIVE ) {
    l_err = reorder_left_bc_bm_bk_bi_cb_kb_mb_right_bc_bn_bk_bj_cb_nb_kb_out_native( m_size_mb_min,
                                                                                     m_size_mb_max,
                                                                                     m_size_nb_min,
                                                                                     m_size_nb_max,
                                                                                     m_size_kb_min,
                                                                                     m_size_kb_max,
                                                                                     i_num_dims_left,
                                                                                     i_num_dims_right,
                                                                                     i_num_dims_out,
                                                                                     i_dim_sizes,
                                                                                     io_dim_ids_left,
                                                                                     io_dim_ids_right,
                                                                                     io_dim_ids_out );
  }

  return l_err;
}

einsum_ir::err_t einsum_ir::backend::BinaryPrimitives::reorder( backend_t                            i_backend_type,
                                                                int64_t                              i_num_dims_left,
                                                                int64_t                              i_num_dims_right,
                                                                int64_t                              i_num_dims_out,
                                                                std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                int64_t                            * io_dim_ids_left,
                                                                int64_t                            * io_dim_ids_right,
                                                                int64_t                            * io_dim_ids_out ) const {
  err_t l_err = err_t::DIMENSION_ORDERING_FAILED;

  // determine dimension types of the tensors
  std::vector< einsum_ir::dim_t > l_dim_types_left;
  std::vector< einsum_ir::dim_t > l_dim_types_right;
  std::vector< einsum_ir::dim_t > l_dim_types_out;
  BinaryContraction::dim_types( i_num_dims_left,
                                i_num_dims_right,
                                i_num_dims_out,
                                io_dim_ids_left,
                                io_dim_ids_right,
                                io_dim_ids_out,
                                &l_dim_types_left,
                                &l_dim_types_right,
                                &l_dim_types_out );

  tenord_t l_tensor_ordering = tenord_t::UNDEFINED_TENORD;

  if( i_backend_type == backend_t::TPP ) {
    l_tensor_ordering = tenord_t::LEFT_BC_BM_BK_BI_KB_MB_CB_RIGHT_BC_BN_BK_BJ_NB_KB_CB_OUT_NATIVE;
  }
  else if( i_backend_type == backend_t::BLAS ) {
    if(    l_dim_types_out.size() > 0
        && l_dim_types_out.back() == einsum_ir::dim_t::C ) {
      l_tensor_ordering = tenord_t::LEFT_BC_BM_BK_BI_CB_KB_MB_RIGHT_BC_BN_BK_BJ_CB_NB_KB_OUT_NATIVE;
    }
    else {
      l_tensor_ordering = tenord_t::LEFT_BC_BM_BK_BI_KB_MB_CB_RIGHT_BC_BN_BK_BJ_NB_KB_CB_OUT_NATIVE;
    }
  }
  else if( i_backend_type == backend_t::TBLIS ) {
    l_tensor_ordering = tenord_t::LEFT_BC_BM_BK_BI_KB_MB_CB_RIGHT_BC_BN_BK_BJ_NB_KB_CB_OUT_NATIVE;
  }
  else {
    l_tensor_ordering = tenord_t::LEFT_NATIVE_RIGHT_NATIVE_OUT_NATIVE;
  }

  l_err = reorder( l_tensor_ordering,
                   i_num_dims_left,
                   i_num_dims_right,
                   i_num_dims_out,
                   i_dim_sizes,
                   io_dim_ids_left,
                   io_dim_ids_right,
                   io_dim_ids_out );

  return l_err;
}