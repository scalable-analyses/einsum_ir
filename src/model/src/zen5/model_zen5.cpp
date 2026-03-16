#include "model_zen5.h"

#include "../common/common.h"
#include "../common/interpolation.h"

namespace einsum_ir::model::zen5 {

  void find_bounds_m(const int* arr, int size, int val, int& idx_lower, double& t) {
    // For values > 128, map them to the 112-128 range based on mod 16
    int search_val = val;
    if (val > 128) {
      int mod16 = val % 16;
      search_val = (mod16 == 0) ? 128 : (112 + mod16);
    }

    const int* exact = std::lower_bound(arr, arr + size, search_val);
    if (exact != arr + size && *exact == search_val) {
      idx_lower = exact - arr;
      t = 0.0;
      return;
    }

    const int* upper = std::upper_bound(arr, arr + size, search_val);

    if (upper == arr) {
      idx_lower = 0;
      t = 0.0;
      return;
    }
    if (upper == arr + size) {
      idx_lower = size - 1;
      t = 0.0;
      return;
    }

    const int* lower = upper - 1;
    idx_lower = lower - arr;

    double v_lower = *lower;
    double v_upper = *upper;
    t = (search_val - v_lower) / (v_upper - v_lower);
  }

  double get_interpolated_gflops(int i_m, int i_n, int i_k, int i_trans_a, int i_trans_b, einsum_ir::model::common::DType i_dtype) {
    (void)i_dtype;
    if (i_trans_a < 0) i_trans_a = 0;
    if (i_trans_a > 1) i_trans_a = 1;
    if (i_trans_b < 0) i_trans_b = 0;
    if (i_trans_b > 1) i_trans_b = 1;

    int m_idx0, n_idx0, k_idx0;
    double t_m, t_n, t_k;

    find_bounds_m(M_VALUES, M_SIZE, i_m, m_idx0, t_m);
    common::find_bounds_with_interpolation(N_VALUES, N_SIZE, i_n, n_idx0, t_n);
    common::find_bounds_with_interpolation(K_VALUES, K_SIZE, i_k, k_idx0, t_k);

    int m_idx1 = (t_m > 0.0 && m_idx0 + 1 < M_SIZE) ? m_idx0 + 1 : m_idx0;
    int n_idx1 = (t_n > 0.0 && n_idx0 + 1 < N_SIZE) ? n_idx0 + 1 : n_idx0;
    int k_idx1 = (t_k > 0.0 && k_idx0 + 1 < K_SIZE) ? k_idx0 + 1 : k_idx0;

    double c000 = gflops_table[m_idx0][n_idx0][k_idx0][i_trans_a][i_trans_b];
    double c100 = gflops_table[m_idx1][n_idx0][k_idx0][i_trans_a][i_trans_b];
    double c010 = gflops_table[m_idx0][n_idx1][k_idx0][i_trans_a][i_trans_b];
    double c110 = gflops_table[m_idx1][n_idx1][k_idx0][i_trans_a][i_trans_b];
    double c001 = gflops_table[m_idx0][n_idx0][k_idx1][i_trans_a][i_trans_b];
    double c101 = gflops_table[m_idx1][n_idx0][k_idx1][i_trans_a][i_trans_b];
    double c011 = gflops_table[m_idx0][n_idx1][k_idx1][i_trans_a][i_trans_b];
    double c111 = gflops_table[m_idx1][n_idx1][k_idx1][i_trans_a][i_trans_b];

    double c00 = common::lerp(c000, c100, t_m);
    double c01 = common::lerp(c001, c101, t_m);
    double c10 = common::lerp(c010, c110, t_m);
    double c11 = common::lerp(c011, c111, t_m);

    double c0 = common::lerp(c00, c10, t_n);
    double c1 = common::lerp(c01, c11, t_n);

    double result = common::lerp(c0, c1, t_k);

    return result;
  }

}  // namespace einsum_ir::model::zen5
