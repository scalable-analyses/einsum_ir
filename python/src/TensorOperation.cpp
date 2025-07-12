#include "TensorOperation.h"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

einsum_ir::binary::exec_t einsum_ir::py::TensorOperation::convert_exec_type(
  einsum_ir::py::TensorOperation::exec_t exec_type
) {
  switch (exec_type) {
    case einsum_ir::py::TensorOperation::exec_t::seq:
      return einsum_ir::binary::exec_t::SEQ;
    case einsum_ir::py::TensorOperation::exec_t::prim:
      return einsum_ir::binary::exec_t::PRIM;
    case einsum_ir::py::TensorOperation::exec_t::shared:
      return einsum_ir::binary::exec_t::OMP;
    case einsum_ir::py::TensorOperation::exec_t::sfc:
      return einsum_ir::binary::exec_t::SFC;
    default:
      break;
  }
  return einsum_ir::binary::exec_t::UNDEFINED_EXECTYPE;
}

einsum_ir::py::TensorOperation::exec_t einsum_ir::py::TensorOperation::convert_exec_type_back(
  einsum_ir::binary::exec_t exec_type
) {
  switch (exec_type) {
    case einsum_ir::binary::exec_t::SEQ:
      return einsum_ir::py::TensorOperation::exec_t::seq;
    case einsum_ir::binary::exec_t::PRIM:
      return einsum_ir::py::TensorOperation::exec_t::prim;
    case einsum_ir::binary::exec_t::OMP:
      return einsum_ir::py::TensorOperation::exec_t::shared;
    case einsum_ir::binary::exec_t::SFC:
      return einsum_ir::py::TensorOperation::exec_t::sfc;
    default:
      break;
  }
  return einsum_ir::py::TensorOperation::exec_t::undefined;
}

einsum_ir::dim_t einsum_ir::py::TensorOperation::convert_dim_type(
  einsum_ir::py::TensorOperation::dim_t dim_type
) {
  switch (dim_type) {
    case einsum_ir::py::TensorOperation::dim_t::c:
      return einsum_ir::dim_t::C;
    case einsum_ir::py::TensorOperation::dim_t::m:
      return einsum_ir::dim_t::M;
    case einsum_ir::py::TensorOperation::dim_t::n:
      return einsum_ir::dim_t::N;
    case einsum_ir::py::TensorOperation::dim_t::k:
      return einsum_ir::dim_t::K;
    default:
      break;
  }
  return einsum_ir::dim_t::UNDEFINED_DIM;
}

einsum_ir::py::TensorOperation::dim_t einsum_ir::py::TensorOperation::convert_dim_type_back(
  einsum_ir::dim_t dim_type
) {
  switch (dim_type) {
    case einsum_ir::dim_t::C:
      return einsum_ir::py::TensorOperation::dim_t::c;
    case einsum_ir::dim_t::M:
      return einsum_ir::py::TensorOperation::dim_t::m;
    case einsum_ir::dim_t::N:
      return einsum_ir::py::TensorOperation::dim_t::n;
    case einsum_ir::dim_t::K:
      return einsum_ir::py::TensorOperation::dim_t::k;
    default:
      break;
  }
  return einsum_ir::py::TensorOperation::dim_t::undefined;
}

einsum_ir::kernel_t einsum_ir::py::TensorOperation::convert_prim_to_kernel(
  einsum_ir::py::TensorOperation::prim_t prim_type
) {
  switch (prim_type) {
    case einsum_ir::py::TensorOperation::prim_t::zero:
      return einsum_ir::kernel_t::ZERO;
    case einsum_ir::py::TensorOperation::prim_t::copy:
      return einsum_ir::kernel_t::COPY;
    case einsum_ir::py::TensorOperation::prim_t::relu:
      return einsum_ir::kernel_t::RELU;
    case einsum_ir::py::TensorOperation::prim_t::gemm:
      return einsum_ir::kernel_t::MADD;
    case einsum_ir::py::TensorOperation::prim_t::brgemm:
      return einsum_ir::kernel_t::BR_MADD;
    default:
      break;
  }
  return einsum_ir::kernel_t::UNDEFINED_KTYPE;
}

std::vector<einsum_ir::binary::iter_property> einsum_ir::py::TensorOperation::create_iter_properties(
    std::vector<dim_t>   const & dim_types,
    std::vector<exec_t>  const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<int64_t> const & strides_in0,
    std::vector<int64_t> const & strides_in1,
    std::vector<int64_t> const & strides_out
) {
  std::vector<einsum_ir::binary::iter_property> iters;
  std::size_t l_num_iters = dim_types.size();
  
  for( std::size_t i = 0; i < l_num_iters; ++i) {
    einsum_ir::binary::iter_property l_iter;
    l_iter.dim_type       = convert_dim_type(dim_types[i]);
    l_iter.exec_type      = convert_exec_type(exec_types[i]);
    l_iter.size           = dim_sizes[i];
    l_iter.stride_left    = strides_in0[i];
    l_iter.stride_right   = strides_in1[i];
    l_iter.stride_out_aux = 0;
    l_iter.stride_out     = strides_out[i];
    iters.push_back(l_iter);
  }
  
  return iters;
}

void einsum_ir::py::TensorOperation::update_parameters_from_iters(
  std::vector<einsum_ir::binary::iter_property> const & iters,
  std::vector<dim_t>                                  & dim_types,
  std::vector<exec_t>                                 & exec_types,
  std::vector<int64_t>                                & dim_sizes,
  std::vector<int64_t>                                & strides_in0,
  std::vector<int64_t>                                & strides_in1,
  std::vector<int64_t>                                & strides_out
) {
  
  size_t num_iters = iters.size();
  dim_types.clear();
  exec_types.clear();
  dim_sizes.clear();
  strides_in0.clear();
  strides_in1.clear();
  strides_out.clear();
  
  dim_types.reserve(num_iters);
  exec_types.reserve(num_iters);
  dim_sizes.reserve(num_iters);
  strides_in0.reserve(num_iters);
  strides_in1.reserve(num_iters);
  strides_out.reserve(num_iters);
  
  for (const auto & l_iter : iters) {
    dim_types.push_back(convert_dim_type_back(l_iter.dim_type));
    exec_types.push_back(convert_exec_type_back(l_iter.exec_type));
    dim_sizes.push_back(l_iter.size);
    strides_in0.push_back(l_iter.stride_left);
    strides_in1.push_back(l_iter.stride_right);
    strides_out.push_back(l_iter.stride_out);
  }
}

einsum_ir::py::TensorOperation::error_t einsum_ir::py::TensorOperation::setup(
  dtype_t                       dtype,
  prim_t                        prim_first,
  prim_t                        prim_main,
  prim_t                        prim_last,
  std::vector< dim_t>   const & dim_types,
  std::vector< exec_t>  const & exec_types,
  std::vector< int64_t> const & dim_sizes,
  std::vector< int64_t> const & strides_in0,
  std::vector< int64_t> const & strides_in1,
  std::vector< int64_t> const & strides_out,
  int64_t                       num_threads
) {
  // backend enums
  std::vector<einsum_ir::dim_t> l_dim_types;
  std::vector<einsum_ir::binary::exec_t> l_exec_types;

  einsum_ir::data_t l_dtype_left  = einsum_ir::data_t::UNDEFINED_DTYPE;
  einsum_ir::data_t l_dtype_right = einsum_ir::data_t::UNDEFINED_DTYPE;
  einsum_ir::data_t l_dtype_comp  = einsum_ir::data_t::UNDEFINED_DTYPE;
  einsum_ir::data_t l_dtype_out   = einsum_ir::data_t::UNDEFINED_DTYPE;

  einsum_ir::kernel_t l_ktype_first = einsum_ir::kernel_t::UNDEFINED_KTYPE;
  einsum_ir::kernel_t l_ktype_main  = einsum_ir::kernel_t::UNDEFINED_KTYPE;
  einsum_ir::kernel_t l_ktype_last  = einsum_ir::kernel_t::UNDEFINED_KTYPE;

  // convert using helper functions
  l_dim_types.reserve(dim_types.size());
  for (const auto & l_dim_type : dim_types) {
    l_dim_types.push_back(convert_dim_type(l_dim_type));
  }

  l_exec_types.reserve(exec_types.size());
  for (const auto & l_exec_type : exec_types) {
    l_exec_types.push_back(convert_exec_type(l_exec_type));
  }

  switch (dtype) {
    case dtype_t::fp32: l_dtype_left = einsum_ir::data_t::FP32; break;
    case dtype_t::fp64: l_dtype_left = einsum_ir::data_t::FP64; break;
    default:            l_dtype_left = einsum_ir::data_t::UNDEFINED_DTYPE; break;
  }
  l_dtype_right = l_dtype_left;
  l_dtype_comp  = l_dtype_left;
  l_dtype_out   = l_dtype_left;

  l_ktype_first = convert_prim_to_kernel(prim_first);
  l_ktype_main = convert_prim_to_kernel(prim_main);
  l_ktype_last = convert_prim_to_kernel(prim_last);

  // dummy strides for auxiliary output tensor
  std::vector<int64_t> strides_out_aux(dim_sizes.size(), 0);

  // set number of threads
  int64_t l_num_threads = num_threads;
#if defined(_OPENMP)
  if (l_num_threads <= 0) {
    l_num_threads = omp_get_max_threads();
  }
#else
  if( l_num_threads <= 0 ) {
    l_num_threads = 1;
  }
#endif

  // init backend
  m_backend.init( l_dim_types,
                  l_exec_types,
                  dim_sizes,
                  strides_in0,
                  strides_in1,
                  strides_out_aux,
                  strides_out,
                  l_dtype_left,
                  l_dtype_right,
                  l_dtype_comp,
                  l_dtype_out,
                  l_ktype_first,
                  l_ktype_main,
                  l_ktype_last,
                  l_num_threads );

  // compile backend 
  einsum_ir::err_t l_err = m_backend.compile();
  if (l_err != einsum_ir::err_t::SUCCESS) {
    return error_t::compilation_failed;
  }

  return error_t::success;
}

void einsum_ir::py::TensorOperation::execute( void const * tensor_in0,
                                              void const * tensor_in1,
                                              void       * tensor_out) {
  m_backend.contract( tensor_in0,
                      tensor_in1,
                      nullptr,
                      tensor_out );
}

einsum_ir::py::TensorOperation::error_t einsum_ir::py::TensorOperation::optimize( dtype_t                dtype,
                                                                                  prim_t               & prim_first,
                                                                                  prim_t               & prim_main,
                                                                                  prim_t               & prim_last,
                                                                                  std::vector<dim_t>   & dim_types,
                                                                                  std::vector<exec_t>  & exec_types,
                                                                                  std::vector<int64_t> & dim_sizes,
                                                                                  std::vector<int64_t> & strides_in0,
                                                                                  std::vector<int64_t> & strides_in1,
                                                                                  std::vector<int64_t> & strides_out,
                                                                                  int64_t                target_m,
                                                                                  int64_t                target_n,
                                                                                  int64_t                target_k,
                                                                                  int64_t                num_threads,
                                                                                  bool                   br_gemm_support,
                                                                                  bool                   packed_gemm_support ) {
  // Create loop properties from input parameters
  std::vector<einsum_ir::binary::iter_property> l_iters = create_iter_properties(
    dim_types,
    exec_types,
    dim_sizes,
    strides_in0,
    strides_in1,
    strides_out
  );
  
  // Convert main primitive type to kernel type  
  einsum_ir::kernel_t l_kernel_main = convert_prim_to_kernel(prim_main);
  
  int64_t l_num_threads = num_threads;
#if defined(_OPENMP)
  if (l_num_threads <= 0) {
    l_num_threads = omp_get_max_threads();
  }
#else
  if( l_num_threads <= 0 ) {
    l_num_threads = 1;
  }
#endif

  // Initialize optimizer
  einsum_ir::binary::ContractionOptimizer l_optimizer;
  l_optimizer.init( &l_iters,
                    &l_kernel_main,
                    l_num_threads,
                    target_m,
                    target_n,
                    target_k,
                    br_gemm_support,
                    packed_gemm_support );
  
  // Run optimization
  l_optimizer.optimize();
  
  // Update output parameters from optimized loops
  update_parameters_from_iters(
    l_iters,
    dim_types,
    exec_types,
    dim_sizes,
    strides_in0,
    strides_in1,
    strides_out
  );
  
  // Update main primitive if kernel type changed (e.g., GEMM -> BR_GEMM)
  if (l_kernel_main == einsum_ir::kernel_t::BR_MADD) {
    prim_main = prim_t::brgemm;
  } else if (l_kernel_main == einsum_ir::kernel_t::MADD) {
    prim_main = prim_t::gemm;
  }
  // Note: other primitive types (first, last) don't change during optimization
  
  return error_t::success;
}
