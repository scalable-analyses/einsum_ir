#include "interpolation.h"

namespace einsum_ir::model::common {

  double lerp(double x0, double x1, double t) {
    return x0 + t * (x1 - x0);
  }

  void find_bounds_with_interpolation(const int* arr, int size, int val, int& idx_lower, double& t) {
    const int* exact = std::lower_bound(arr, arr + size, val);
    if (exact != arr + size && *exact == val) {
      idx_lower = exact - arr;
      t = 0.0;
      return;
    }

    const int* upper = std::upper_bound(arr, arr + size, val);

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
    t = (val - v_lower) / (v_upper - v_lower);
  }

}  // namespace einsum_ir::model::common
