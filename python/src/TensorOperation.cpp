#include "TensorOperation.h"
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

einsum_ir::py::TensorOperation::op_type_t
einsum_ir::py::TensorOperation::determine_op_type( prim_t prim_main ) {
  switch (prim_main) {
    case prim_t::zero:
    case prim_t::copy:
      return op_type_t::unary;

    case prim_t::gemm:
    case prim_t::brgemm:
      return op_type_t::binary;

    default:
      return op_type_t::undefined;
  }
}

int64_t einsum_ir::py::TensorOperation::get_num_threads( int64_t num_threads ) {
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
  return l_num_threads;
}

einsum_ir::basic::exec_t einsum_ir::py::TensorOperation::convert_exec_type(
  einsum_ir::py::TensorOperation::exec_t exec_type
) {
  switch (exec_type) {
    case einsum_ir::py::TensorOperation::exec_t::seq:
      return einsum_ir::basic::exec_t::SEQ;
    case einsum_ir::py::TensorOperation::exec_t::prim:
      return einsum_ir::basic::exec_t::PRIM;
    case einsum_ir::py::TensorOperation::exec_t::shared:
      return einsum_ir::basic::exec_t::OMP;
    case einsum_ir::py::TensorOperation::exec_t::sfc:
      return einsum_ir::basic::exec_t::SFC;
    default:
      break;
  }
  return einsum_ir::basic::exec_t::UNDEFINED_EXECTYPE;
}

einsum_ir::py::TensorOperation::exec_t einsum_ir::py::TensorOperation::convert_exec_type_back(
  einsum_ir::basic::exec_t exec_type
) {
  switch (exec_type) {
    case einsum_ir::basic::exec_t::SEQ:
      return einsum_ir::py::TensorOperation::exec_t::seq;
    case einsum_ir::basic::exec_t::PRIM:
      return einsum_ir::py::TensorOperation::exec_t::prim;
    case einsum_ir::basic::exec_t::OMP:
      return einsum_ir::py::TensorOperation::exec_t::shared;
    case einsum_ir::basic::exec_t::SFC:
      return einsum_ir::py::TensorOperation::exec_t::sfc;
    default:
      break;
  }
  return einsum_ir::py::TensorOperation::exec_t::undefined;
}

einsum_ir::basic::dim_t einsum_ir::py::TensorOperation::convert_dim_type(
  einsum_ir::py::TensorOperation::dim_t dim_type
) {
  switch (dim_type) {
    case einsum_ir::py::TensorOperation::dim_t::c:
      return einsum_ir::basic::dim_t::C;
    case einsum_ir::py::TensorOperation::dim_t::m:
      return einsum_ir::basic::dim_t::M;
    case einsum_ir::py::TensorOperation::dim_t::n:
      return einsum_ir::basic::dim_t::N;
    case einsum_ir::py::TensorOperation::dim_t::k:
      return einsum_ir::basic::dim_t::K;
    default:
      break;
  }
  return einsum_ir::basic::dim_t::UNDEFINED_DIM;
}

einsum_ir::py::TensorOperation::dim_t einsum_ir::py::TensorOperation::convert_dim_type_back(
  einsum_ir::basic::dim_t dim_type
) {
  switch (dim_type) {
    case einsum_ir::basic::dim_t::C:
      return einsum_ir::py::TensorOperation::dim_t::c;
    case einsum_ir::basic::dim_t::M:
      return einsum_ir::py::TensorOperation::dim_t::m;
    case einsum_ir::basic::dim_t::N:
      return einsum_ir::py::TensorOperation::dim_t::n;
    case einsum_ir::basic::dim_t::K:
      return einsum_ir::py::TensorOperation::dim_t::k;
    default:
      break;
  }
  return einsum_ir::py::TensorOperation::dim_t::undefined;
}

einsum_ir::basic::kernel_t einsum_ir::py::TensorOperation::convert_prim_to_kernel(
  einsum_ir::py::TensorOperation::prim_t prim_type
) {
  switch (prim_type) {
    case einsum_ir::py::TensorOperation::prim_t::zero:
      return einsum_ir::basic::kernel_t::ZERO;
    case einsum_ir::py::TensorOperation::prim_t::copy:
      return einsum_ir::basic::kernel_t::COPY;
    case einsum_ir::py::TensorOperation::prim_t::relu:
      return einsum_ir::basic::kernel_t::RELU;
    case einsum_ir::py::TensorOperation::prim_t::gemm:
      return einsum_ir::basic::kernel_t::MADD;
    case einsum_ir::py::TensorOperation::prim_t::brgemm:
      return einsum_ir::basic::kernel_t::BR_MADD;
    default:
      break;
  }
  return einsum_ir::basic::kernel_t::UNDEFINED_KTYPE;
}

std::vector<einsum_ir::basic::iter_property> einsum_ir::py::TensorOperation::create_iter_properties(
    std::vector<dim_t>   const & dim_types,
    std::vector<exec_t>  const & exec_types,
    std::vector<int64_t> const & dim_sizes,
    std::vector<int64_t> const & strides_in0,
    std::vector<int64_t> const & strides_in1,
    std::vector<int64_t> const & strides_out
) {
  std::vector<einsum_ir::basic::iter_property> iters;
  std::size_t l_num_iters = dim_types.size();
  
  for( std::size_t i = 0; i < l_num_iters; ++i) {
    einsum_ir::basic::iter_property l_iter;
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
  std::vector<einsum_ir::basic::iter_property> const & iters,
  std::vector<dim_t>                                 & dim_types,
  std::vector<exec_t>                                & exec_types,
  std::vector<int64_t>                               & dim_sizes,
  std::vector<int64_t>                               & strides_in0,
  std::vector<int64_t>                               & strides_in1,
  std::vector<int64_t>                               & strides_out
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

int64_t einsum_ir::py::TensorOperation::dtype_to_num_bytes( dtype_t dtype ) {
  switch (dtype) {
    case dtype_t::fp32: return sizeof(float);
    case dtype_t::fp64: return sizeof(double);
    default:            return 0; // Undefined or unsupported type
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
  int64_t l_num_threads[3] = {1, 1, 1};
  l_num_threads[0] = get_num_threads(num_threads);

  m_op_type = determine_op_type(prim_main);

  if (m_op_type == op_type_t::undefined) {
    return error_t::compilation_failed;
  }

  if (m_op_type == op_type_t::unary) {
    // Validate: prim_first and prim_last must be 'none'
    if (prim_first != prim_t::none || prim_last != prim_t::none) {
      return error_t::compilation_failed;
    }

    return setup_unary(dtype, prim_main, exec_types,
                       dim_sizes, strides_in0, strides_out, l_num_threads[0]);
  }
  else { // binary
    return setup_binary(dtype, prim_first, prim_main, prim_last,
                        dim_types, exec_types, dim_sizes,
                        strides_in0, strides_in1, strides_out, l_num_threads);
  }
}

einsum_ir::py::TensorOperation::error_t einsum_ir::py::TensorOperation::setup_unary(
  dtype_t                       dtype,
  prim_t                        prim_main,
  std::vector< exec_t>  const & exec_types,
  std::vector< int64_t> const & dim_sizes,
  std::vector< int64_t> const & strides_in0,
  std::vector< int64_t> const & strides_out,
  int64_t                       num_threads
) {
  // Convert data types
  einsum_ir::basic::data_t l_dtype_in = einsum_ir::basic::data_t::UNDEFINED_DTYPE;
  switch (dtype) {
    case dtype_t::fp32: l_dtype_in = einsum_ir::basic::data_t::FP32; break;
    case dtype_t::fp64: l_dtype_in = einsum_ir::basic::data_t::FP64; break;
    default:            l_dtype_in = einsum_ir::basic::data_t::UNDEFINED_DTYPE; break;
  }
  einsum_ir::basic::data_t l_dtype_comp = l_dtype_in;
  einsum_ir::basic::data_t l_dtype_out  = l_dtype_in;

  // Convert kernel type
  einsum_ir::basic::kernel_t l_ktype_main = convert_prim_to_kernel(prim_main);

  // Convert exec types
  std::vector<einsum_ir::basic::exec_t> l_exec_types;
  l_exec_types.reserve(exec_types.size());
  for (const auto & l_exec_type : exec_types) {
    l_exec_types.push_back(convert_exec_type(l_exec_type));
  }

  // Initialize unary backend
  m_backend_unary.init(l_exec_types, dim_sizes, strides_in0, strides_out,
                       l_dtype_in, l_dtype_comp, l_dtype_out,
                       l_ktype_main, num_threads);

  // Compile backend
  einsum_ir::basic::err_t l_err = m_backend_unary.compile();
  if (l_err != einsum_ir::basic::err_t::SUCCESS) {
    return error_t::compilation_failed;
  }

  return error_t::success;
}

einsum_ir::py::TensorOperation::error_t einsum_ir::py::TensorOperation::setup_binary(
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
  int64_t                       num_threads[3]
) {
  // backend enums
  std::vector<einsum_ir::basic::dim_t> l_dim_types;
  std::vector<einsum_ir::basic::exec_t> l_exec_types;

  einsum_ir::basic::data_t l_dtype_left  = einsum_ir::basic::data_t::UNDEFINED_DTYPE;
  einsum_ir::basic::data_t l_dtype_right = einsum_ir::basic::data_t::UNDEFINED_DTYPE;
  einsum_ir::basic::data_t l_dtype_comp  = einsum_ir::basic::data_t::UNDEFINED_DTYPE;
  einsum_ir::basic::data_t l_dtype_out   = einsum_ir::basic::data_t::UNDEFINED_DTYPE;

  einsum_ir::basic::kernel_t l_ktype_first = einsum_ir::basic::kernel_t::UNDEFINED_KTYPE;
  einsum_ir::basic::kernel_t l_ktype_main  = einsum_ir::basic::kernel_t::UNDEFINED_KTYPE;
  einsum_ir::basic::kernel_t l_ktype_last  = einsum_ir::basic::kernel_t::UNDEFINED_KTYPE;

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
    case dtype_t::fp32: l_dtype_left = einsum_ir::basic::data_t::FP32; break;
    case dtype_t::fp64: l_dtype_left = einsum_ir::basic::data_t::FP64; break;
    default:            l_dtype_left = einsum_ir::basic::data_t::UNDEFINED_DTYPE; break;
  }
  l_dtype_right = l_dtype_left;
  l_dtype_comp  = l_dtype_left;
  l_dtype_out   = l_dtype_left;

  l_ktype_first = convert_prim_to_kernel(prim_first);
  l_ktype_main = convert_prim_to_kernel(prim_main);
  l_ktype_last = convert_prim_to_kernel(prim_last);

  // dummy strides for auxiliary output tensor
  std::vector<int64_t> strides_out_aux(dim_sizes.size(), 0);

  // dummy strides for packing
  std::vector<int64_t> packing_strides_left(dim_sizes.size(), 0);
  std::vector<int64_t> packing_strides_right(dim_sizes.size(), 0);

  // init backend
  m_backend_binary.init( l_dim_types,
                         l_exec_types,
                         dim_sizes,
                         strides_in0,
                         strides_in1,
                         strides_out_aux,
                         strides_out,
                         packing_strides_left,
                         packing_strides_right,
                         l_dtype_left,
                         l_dtype_right,
                         l_dtype_comp,
                         l_dtype_out,
                         l_ktype_first,
                         l_ktype_main,
                         l_ktype_last,
                         num_threads[0],
                         num_threads[1],
                         num_threads[2],
                         nullptr );

  // compile backend 
  einsum_ir::basic::err_t l_err = m_backend_binary.compile();
  if (l_err != einsum_ir::basic::err_t::SUCCESS) {
    return error_t::compilation_failed;
  }

  return error_t::success;
}

void einsum_ir::py::TensorOperation::execute( void const * tensor_in0,
                                              void const * tensor_in1,
                                              void       * tensor_out) {
  if (m_op_type == op_type_t::unary) {
    m_backend_unary.eval(tensor_in0, tensor_out);
  }
  else if (m_op_type == op_type_t::binary) {
    m_backend_binary.contract(tensor_in0, tensor_in1, nullptr, tensor_out);
  }
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
                                                                                  bool                   packed_gemm_support,
                                                                                  int64_t                l2_cache_size ) {
  int64_t l_num_threads[3] = {1, 1, 1};
  l_num_threads[0] = get_num_threads(num_threads);

  op_type_t op_type = determine_op_type(prim_main);

  if (op_type == op_type_t::undefined) {
    return error_t::compilation_failed;
  }

  if (op_type == op_type_t::unary) {
    return optimize_unary(dtype, prim_main, dim_types, exec_types,
                          dim_sizes, strides_in0, strides_out,
                          l_num_threads[0]);
  }
  else { // binary
    return optimize_binary(dtype, prim_first, prim_main, prim_last,
                           dim_types, exec_types, dim_sizes,
                           strides_in0, strides_in1, strides_out,
                           target_m, target_n, target_k, l_num_threads,
                           br_gemm_support, packed_gemm_support, l2_cache_size);
  }
}

einsum_ir::py::TensorOperation::error_t einsum_ir::py::TensorOperation::optimize_unary(
  dtype_t                  dtype,
  prim_t                 & prim_main,
  std::vector< dim_t >   & dim_types,
  std::vector< exec_t >  & exec_types,
  std::vector< int64_t > & dim_sizes,
  std::vector< int64_t > & strides_in0,
  std::vector< int64_t > & strides_out,
  int64_t                  num_threads
) {
  // Create iter_properties for unary operation
  std::vector<einsum_ir::basic::iter_property> l_iters;
  l_iters.reserve(dim_sizes.size());

  for (size_t i = 0; i < dim_sizes.size(); ++i) {
    einsum_ir::basic::iter_property l_iter;
    l_iter.dim_type       = convert_dim_type(dim_types[i]);
    l_iter.exec_type      = convert_exec_type(exec_types[i]);
    l_iter.size           = dim_sizes[i];
    l_iter.stride_left    = strides_in0[i];
    l_iter.stride_right   = 0;  // unused for unary
    l_iter.stride_out_aux = 0;  // unused for unary
    l_iter.stride_out     = strides_out[i];
    l_iter.packing_stride_left  = 0;  // unused for unary
    l_iter.packing_stride_right = 0;  // unused for unary
    l_iters.push_back(l_iter);
  }

  // Initialize optimizer (scalar_optim = false)
  einsum_ir::basic::UnaryOptimizer l_optimizer;
  l_optimizer.init(&l_iters, num_threads, false);

  // Run optimization
  einsum_ir::basic::err_t l_err = l_optimizer.optimize();
  if (l_err != einsum_ir::basic::err_t::SUCCESS) {
    return error_t::compilation_failed;
  }

  // Update parameters from optimized iters
  dim_types.clear();
  exec_types.clear();
  dim_sizes.clear();
  strides_in0.clear();
  strides_out.clear();

  dim_types.reserve(l_iters.size());
  exec_types.reserve(l_iters.size());
  dim_sizes.reserve(l_iters.size());
  strides_in0.reserve(l_iters.size());
  strides_out.reserve(l_iters.size());

  for (const auto & l_iter : l_iters) {
    dim_types.push_back(convert_dim_type_back(l_iter.dim_type));
    exec_types.push_back(convert_exec_type_back(l_iter.exec_type));
    dim_sizes.push_back(l_iter.size);
    strides_in0.push_back(l_iter.stride_left);
    strides_out.push_back(l_iter.stride_out);
  }

  return error_t::success;
}

einsum_ir::py::TensorOperation::error_t einsum_ir::py::TensorOperation::optimize_binary(
  dtype_t                  dtype,
  prim_t                 & prim_first,
  prim_t                 & prim_main,
  prim_t                 & prim_last,
  std::vector< dim_t >   & dim_types,
  std::vector< exec_t >  & exec_types,
  std::vector< int64_t > & dim_sizes,
  std::vector< int64_t > & strides_in0,
  std::vector< int64_t > & strides_in1,
  std::vector< int64_t > & strides_out,
  int64_t                  target_m,
  int64_t                  target_n,
  int64_t                  target_k,
  int64_t                  num_threads[3],
  bool                     br_gemm_support,
  bool                     packed_gemm_support,
  int64_t                  l2_cache_size
) {
  // Create loop properties from input parameters
  std::vector<einsum_ir::basic::iter_property> l_iters = create_iter_properties(
    dim_types,
    exec_types,
    dim_sizes,
    strides_in0,
    strides_in1,
    strides_out
  );
  
  // Convert main primitive type to kernel type  
  einsum_ir::basic::kernel_t l_kernel_main = convert_prim_to_kernel(prim_main);

  int64_t l_num_bytes = dtype_to_num_bytes(dtype);

  einsum_ir::basic::packed_gemm_t l_packed_gemm_support =
    packed_gemm_support ? einsum_ir::basic::packed_gemm_t::NONE
                        : einsum_ir::basic::packed_gemm_t::ALL_STRIDE_ONE;

  // Initialize optimizer
  einsum_ir::basic::ContractionOptimizer l_optimizer;
  l_optimizer.init( &l_iters,
                    &l_kernel_main,
                    target_m,
                    target_n,
                    target_k,
                    false, // TODO: Add SFC support flag
                    br_gemm_support,
                    false, // TODO: Add packing support flag
                    l_packed_gemm_support,
                    l_num_bytes,
                    l2_cache_size,
                    &num_threads[0],
                    &num_threads[1],
                    &num_threads[2] );
  
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
  if (l_kernel_main == einsum_ir::basic::kernel_t::BR_MADD) {
    prim_main = prim_t::brgemm;
  } else if (l_kernel_main == einsum_ir::basic::kernel_t::MADD) {
    prim_main = prim_t::gemm;
  }
  // Note: other primitive types (first, last) don't change during optimization
  
  return error_t::success;
}
