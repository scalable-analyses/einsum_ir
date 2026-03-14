#include "model_a76.h"

#include "../common/common.h"
#include "../common/interpolation.h"

namespace einsum_ir::model::a76 {

  int get_exact_index(const int* arr, int size, int val) {
    const int* exact = std::lower_bound(arr, arr + size, val);
    if (exact != arr + size && *exact == val) {
      return exact - arr;
    }

    if (val > arr[size - 1]) {
      return size - 1;
    }

    if (val < arr[0]) {
      return 0;
    }

    int closest_idx = 0;
    int min_diff = std::abs(val - arr[0]);
    for (int i = 1; i < size; i++) {
      int diff = std::abs(val - arr[i]);
      if (diff < min_diff) {
        min_diff = diff;
        closest_idx = i;
      }
    }
    return closest_idx;
  }

  double get_gflops(int i_m, int i_n, int i_k, int i_trans_a, int i_trans_b) {
    if (i_trans_a < 0) i_trans_a = 0;
    if (i_trans_a > 1) i_trans_a = 1;
    if (i_trans_b < 0) i_trans_b = 0;
    if (i_trans_b > 1) i_trans_b = 1;

    int m_idx = get_exact_index(M_VALUES, M_SIZE, i_m);
    int n_idx = get_exact_index(N_VALUES, N_SIZE, i_n);

    int k_idx0;
    double t_k;
    common::find_bounds_with_interpolation(K_VALUES, K_SIZE, i_k, k_idx0, t_k);
    int k_idx1 = (t_k > 0.0 && k_idx0 + 1 < K_SIZE) ? k_idx0 + 1 : k_idx0;

    double c0 = gflops_table[m_idx][n_idx][k_idx0][i_trans_a][i_trans_b];
    double c1 = gflops_table[m_idx][n_idx][k_idx1][i_trans_a][i_trans_b];

    double result = common::lerp(c0, c1, t_k);

    return result;
  }

  void get_blocking(int64_t i_m,
                    int64_t i_n,
                    int64_t i_k,
                    int i_transpose_a,
                    int i_transpose_b,
                    jit_sizes& kernels) {
    int64_t l_m_full = 0;
    int64_t l_m_full_count = 0;
    int64_t l_m_rest = 0;

    int64_t l_n_first = 0;
    int64_t l_n_first_count = 0;
    int64_t l_n_second = 0;
    int64_t l_n_second_count = 0;

    int l_m_blocking = 16;

    if (i_m >= l_m_blocking) {
      l_m_full = l_m_blocking;
      l_m_full_count = i_m / l_m_blocking;
      l_m_rest = i_m % l_m_blocking;
    } else {
      l_m_rest = i_m;
    }

    int64_t l_m_registers = (l_m_full_count > 0) ? (l_m_full + 3) / 4 : (l_m_rest + 3) / 4;
    int64_t l_n_registers = (l_n_first + 3) / 4;

    l_n_first = i_n;

    while (l_m_registers * l_n_first + l_m_registers + l_n_first > 32) {
      l_n_first--;
    }

    unsigned int l_number_of_chunks = ((i_n - 1) / l_n_first) + 1;
    unsigned int l_modulo = i_n % l_number_of_chunks;
    unsigned int l_n2 = i_n / l_number_of_chunks;
    unsigned int l_n1 = l_n2 + 1;
    unsigned int l_N2 = 0;
    unsigned int l_N1 = 0;
    unsigned int l_chunk = 0;
    unsigned int l_ret = 0;

    if (l_n1 > l_n_first)
      l_n1 = l_n_first;
    for (l_chunk = 0; l_chunk < l_number_of_chunks; ++l_chunk) {
      if (l_chunk < l_modulo) {
        l_N1 += l_n1;
      } else {
        l_N2 += l_n2;
      }
    }

    if (l_modulo == 0) {
      l_n1 = l_n2;
      l_N1 = l_N2;
      l_n2 = 0;
      l_N2 = 0;
    }

    if ((l_N1 % l_n1) != 0) {
      l_ret = 1;
    }
    if (l_n2 != 0) {
      if (l_N2 % l_n2 != 0) {
        l_ret = 1;
      }
    }
    if (l_ret != 0) {
      std::cerr << "Blocking isnt working correct!!" << std::endl;
    }

    l_n_first_count = l_N1 / l_n1;
    if (l_n2 > 0) {
      l_n_second_count = l_N2 / l_n2;
    } else {
      l_n_second_count = 0;
    }

    l_n_first = l_n1;
    l_n_second = l_n2;

    kernels.k1.m = l_m_full;
    kernels.k1.n = l_n_first;
    if (l_m_rest > 0) {
      kernels.k2.m = l_m_rest;
      kernels.k2.n = l_n_first;
    } else {
      kernels.k2.m = 0;
      kernels.k2.n = 0;
    }

    if (l_n_second > 0) {
      kernels.k3.m = l_m_full;
      kernels.k3.n = l_n_second;
    } else {
      kernels.k3.m = 0;
      kernels.k3.n = 0;
    }

    if (l_m_rest > 0 && l_n_second > 0) {
      kernels.k4.m = l_m_rest;
      kernels.k4.n = l_n_second;
    } else {
      kernels.k4.m = 0;
      kernels.k4.n = 0;
    }

    kernels.first_n_count = l_n_first_count;
    kernels.second_n_count = l_n_second_count;

    kernels.m_count_full_count = i_m / 16;
  }

  double calculate_gflops(int i_m,
                          int i_n,
                          int i_k,
                          jit_sizes& kernels,
                          int i_transpose_a,
                          int i_transpose_b) {
    double flops_area_a = 2.0 * (i_m - kernels.k2.m) * (kernels.k1.n * kernels.first_n_count) * i_k;
    double flops_area_b = 2.0 * (kernels.k2.m) * (kernels.k1.n * kernels.first_n_count) * i_k;
    double flops_area_c = 2.0 * (i_m - kernels.k2.m) * (kernels.k3.n * kernels.second_n_count) * i_k;
    double flops_area_d = 2.0 * (kernels.k2.m) * (kernels.k3.n * kernels.second_n_count) * i_k;
    double gesamt_flops = flops_area_a + flops_area_b + flops_area_c + flops_area_d;

    double gflops_part_k1 = 0.0;
    double gflops_part_k2 = 0.0;
    double gflops_part_k3 = 0.0;
    double gflops_part_k4 = 0.0;

    double time_1_area_k1 = 0.0;
    double time_1_area_k2 = 0.0;
    double time_1_area_k3 = 0.0;
    double time_1_area_k4 = 0.0;

    if (gesamt_flops != (i_m * i_n * i_k * 2.0)) {
      std::cerr << "Flops calculation is wrong!" << std::endl;
      std::cout << gesamt_flops << " != " << (i_m * i_n * i_k * 2.0) << std::endl;
    }

    if (kernels.k1.m > 0 && kernels.k1.n > 0) {
      gflops_part_k1 = get_gflops(kernels.k1.m, kernels.k1.n, i_k, i_transpose_a, i_transpose_b);
    }
    if (kernels.k2.m > 0 && kernels.k2.n > 0) {
      gflops_part_k2 = get_gflops(kernels.k2.m, kernels.k2.n, i_k, i_transpose_a, i_transpose_b);
    }
    if (kernels.k3.m > 0 && kernels.k3.n > 0) {
      gflops_part_k3 = get_gflops(kernels.k3.m, kernels.k3.n, i_k, i_transpose_a, i_transpose_b);
    }
    if (kernels.k4.m > 0 && kernels.k4.n > 0) {
      gflops_part_k4 = get_gflops(kernels.k4.m, kernels.k4.n, i_k, i_transpose_a, i_transpose_b);
    }

    if (gflops_part_k1 > 0.0) {
      time_1_area_k1 = flops_area_a / (gflops_part_k1);
    }
    if (gflops_part_k2 > 0.0) {
      time_1_area_k2 = flops_area_b / (gflops_part_k2);
    }
    if (gflops_part_k3 > 0.0) {
      time_1_area_k3 = flops_area_c / (gflops_part_k3);
    }
    if (gflops_part_k4 > 0.0) {
      time_1_area_k4 = flops_area_d / (gflops_part_k4);
    }

    double time_total = time_1_area_k1 + time_1_area_k2 + time_1_area_k3 + time_1_area_k4;
    time_total /= 1e9;

    double total_perf_flops = 1 / time_total;
    total_perf_flops *= gesamt_flops;
    total_perf_flops /= 1e9;

    return total_perf_flops;
  }

  double get_interpolated_gflops(int i_m,
                                 int i_n,
                                 int i_k,
                                 int i_transpose_a,
                                 int i_transpose_b,
                                 einsum_ir::model::common::DType i_dtype) {
    (void)i_dtype;
    jit_sizes kernels;

    get_blocking(i_m,
                 i_n,
                 i_k,
                 i_transpose_a,
                 i_transpose_b,
                 kernels);

    return calculate_gflops(i_m, i_n, i_k, kernels, i_transpose_a, i_transpose_b);
  }
}  // namespace einsum_ir::model::a76