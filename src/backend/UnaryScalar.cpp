#include "UnaryScalar.h"

template < typename T >
void einsum_ir::backend::UnaryScalar::kernel_copy( void const * i_data_src,
                                                   void       * io_data_dst ) {
  T const * l_data_src = (T const *) i_data_src;
  T * l_data_dst = (T *) io_data_dst;

  *l_data_dst = *l_data_src;
}

einsum_ir::err_t einsum_ir::backend::UnaryScalar::compile() {
  Unary::compile_base();

  // determine if all dtypes are FP32 or FP64
  bool l_dtype_all_fp32 = false;
  bool l_dtype_all_fp64 = false;

  if(    m_dtype_in   == FP32
      && m_dtype_comp == FP32
      && m_dtype_out  == FP32 ) {
    l_dtype_all_fp32 = true;
  }
  else if(    m_dtype_in   == FP64
           && m_dtype_comp == FP64
           && m_dtype_out  == FP64 ) {
    l_dtype_all_fp64 = true;
  }

  // assign kernel
  if( m_ktype_main == COPY ) {
    if( l_dtype_all_fp32 ) {
      m_kernel_main = &kernel_copy< float >;
    }
    else if( l_dtype_all_fp64 ) {
      m_kernel_main = &kernel_copy< double >;
    }
    else {
      return einsum_ir::COMPILATION_FAILED;
    }
  }
  else {
    return einsum_ir::COMPILATION_FAILED;
  }

  // init and compile loop interface
  m_unary_loops.init( m_num_dims,
                      m_sizes_out.data(),
                      m_strides_in.data(),
                      m_strides_out.data(),
                      ce_n_bytes( m_dtype_in ),
                      ce_n_bytes( m_dtype_out ),
                      m_kernel_main );

  einsum_ir::err_t l_err = m_unary_loops.compile();
  if( l_err != einsum_ir::SUCCESS ) {
    return einsum_ir::COMPILATION_FAILED;
  }

  m_compiled = true;

  return err_t::SUCCESS;
}

void einsum_ir::backend::UnaryScalar::threading( int64_t i_num_tasks_target  ) {
  m_unary_loops.threading( i_num_tasks_target );
}

void einsum_ir::backend::UnaryScalar::eval( void const * i_tensor_in,
                                            void       * io_tensor_out ) {
  m_unary_loops.eval( i_tensor_in,
                      io_tensor_out );
}