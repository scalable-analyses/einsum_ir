#ifndef EINSUM_IR_MODEL_COMMON_INTERPOLATION_H
#define EINSUM_IR_MODEL_COMMON_INTERPOLATION_H

#include <algorithm>

namespace einsum_ir::model::common {

  /**
   * Linear interpolation between two values.
   *
   * @param x0 The first value.
   * @param x1 The second value.
   * @param t The interpolation factor (0.0 to 1.0).
   * @return The interpolated value.
   */
  double lerp(double x0,
              double x1,
              double t);

  /**
   * Find surrounding indices and interpolation factor for a value in an array.
   * Performs linear interpolation between array elements.
   *
   * @param arr The sorted array to search.
   * @param size The size of the array.
   * @param val The value to find.
   * @param idx_lower Output: the lower index for interpolation.
   * @param t Output: the interpolation factor (0.0 to 1.0).
   */
  void find_bounds_with_interpolation(const int* arr,
                                      int size,
                                      int val,
                                      int& idx_lower,
                                      double& t);

}  // namespace einsum_ir::model::common

#endif  // EINSUM_IR_MODEL_COMMON_INTERPOLATION_H
