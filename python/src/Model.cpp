#include "Model.h"

namespace einsum_ir {
  namespace py {

    Model::Model(model_t model_type,
                 double peak_gflops,
                 int vector_size)
        : m_model_type(model_type),
          m_peak_gflops(peak_gflops),
          m_vector_size(vector_size) {
    }

    void Model::extract_primitive_dims(
        prim_t prim_main,
        std::vector<dim_t> const& dim_types,
        std::vector<exec_t> const& exec_types,
        std::vector<int64_t> const& dim_sizes,
        int64_t& o_m, int64_t& o_n, int64_t& o_k, int64_t& o_br) {

      std::vector<int64_t> prim_dims;
      for (size_t i = 0; i < dim_types.size(); i++) {
        if (exec_types[i] == exec_t::prim) {
          prim_dims.push_back(dim_sizes[i]);
        }
      }

      bool is_brgemm = (prim_main == prim_t::brgemm);

      if (is_brgemm) {
        o_br = prim_dims[0];
        o_m = prim_dims[1];
        o_n = prim_dims[2];
        o_k = prim_dims[3];
      } else {
        o_m = prim_dims[0];
        o_n = prim_dims[1];
        o_k = prim_dims[2];
        o_br = 1;
      }
    }

    void Model::extract_transpose_flags(
        std::vector<dim_t> const& dim_types,
        std::vector<exec_t> const& exec_types,
        std::vector<std::vector<std::vector<int64_t>>> const& strides,
        bool& o_trans_a, bool& o_trans_b) {

      auto const& strides_a = strides[0][0];
      auto const& strides_b = strides[0][1];

      int64_t m_idx = -1, n_idx = -1, k_idx = -1;

      for (size_t i = 0; i < dim_types.size(); i++) {
        if (dim_types[i] == dim_t::m && exec_types[i] == exec_t::prim) {
          m_idx = i;
        } else if (dim_types[i] == dim_t::n && exec_types[i] == exec_t::prim) {
          n_idx = i;
        } else if (dim_types[i] == dim_t::k && exec_types[i] == exec_t::prim) {
          k_idx = i;
        }
      }

      int64_t stride_a_k = strides_a[k_idx];
      int64_t stride_b_n = strides_b[n_idx];

      o_trans_a = (stride_a_k == 1);
      o_trans_b = (stride_b_n == 1);
    }

    int64_t Model::compute_gemm_iter(
        std::vector<exec_t> const& exec_types,
        std::vector<int64_t> const& dim_sizes) {
      int64_t gemm_iter = 1;
      for (size_t i = 0; i < exec_types.size(); i++) {
        if (exec_types[i] != exec_t::prim) {
          gemm_iter *= dim_sizes[i];
        }
      }
      return gemm_iter;
    }

    einsum_ir::model::common::Model Model::convert_model_type() const {
      switch (m_model_type) {
        case model_t::zen5:
          return einsum_ir::model::common::Model::ZEN5;
        case model_t::m4:
          return einsum_ir::model::common::Model::M4;
        case model_t::a76:
          return einsum_ir::model::common::Model::A76;
        case model_t::generic:
        default:
          return einsum_ir::model::common::Model::GENERIC;
      }
    }

    einsum_ir::model::common::DType Model::convert_dtype(dtype_t dtype) {
      switch (dtype) {
        case dtype_t::fp32:
          return einsum_ir::model::common::DType::FP32;
        case dtype_t::fp64:
          return einsum_ir::model::common::DType::FP64;
        default:
          return einsum_ir::model::common::DType::FP32;
      }
    }

    double Model::predict(prim_t prim_main,
                          std::vector<dim_t> const& dim_types,
                          std::vector<exec_t> const& exec_types,
                          std::vector<int64_t> const& dim_sizes,
                          std::vector<std::vector<std::vector<int64_t>>> const& strides,
                          dtype_t dtype) const {
      // Extract configuration
      int64_t m, n, k, br;
      extract_primitive_dims(prim_main, dim_types, exec_types, dim_sizes, m, n, k, br);

      bool trans_a, trans_b;
      extract_transpose_flags(dim_types, exec_types, strides, trans_a, trans_b);

      int64_t gemm_iter = compute_gemm_iter(exec_types, dim_sizes);

      // Perform prediction
      double o_gflops = 0.0;
      double time_per_gemm = einsum_ir::model::common::get_time_model( static_cast<int>(m),
                                                                       static_cast<int>(n),
                                                                       static_cast<int>(k * br),
                                                                       trans_a ? 1 : 0,
                                                                       trans_b ? 1 : 0,
                                                                       convert_dtype(dtype),
                                                                       convert_model_type(),
                                                                       o_gflops,
                                                                       m_peak_gflops,
                                                                       m_vector_size);

      return time_per_gemm * static_cast<double>(gemm_iter);
    }

    double Model::predict_gflops(prim_t prim_main,
                                 std::vector<dim_t> const& dim_types,
                                 std::vector<exec_t> const& exec_types,
                                 std::vector<int64_t> const& dim_sizes,
                                 std::vector<std::vector<std::vector<int64_t>>> const& strides,
                                 dtype_t dtype) const {
      // Extract configuration
      int64_t m, n, k, br;
      extract_primitive_dims(prim_main, dim_types, exec_types, dim_sizes, m, n, k, br);

      bool trans_a, trans_b;
      extract_transpose_flags(dim_types, exec_types, strides, trans_a, trans_b);

      // Perform prediction
      double o_gflops = 0.0;
      einsum_ir::model::common::get_time_model( static_cast<int>(m),
                                                static_cast<int>(n),
                                                static_cast<int>(k * br),
                                                trans_a ? 1 : 0,
                                                trans_b ? 1 : 0,
                                                convert_dtype(dtype),
                                                convert_model_type(),
                                                o_gflops,
                                                m_peak_gflops,
                                                m_vector_size );

      return o_gflops;
    }

  }  // namespace py
}  // namespace einsum_ir
