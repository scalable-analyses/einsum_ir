#ifndef EINSUM_IR_MODEL_COMMON_COMMON_H
#define EINSUM_IR_MODEL_COMMON_COMMON_H

#include <chrono>
#include <iostream>
#include <random>

#include "a76/model_a76.h"
#include "generic/model_generic.h"
#include "m4/model_m4.h"
#include "zen5/model_zen5.h"

namespace einsum_ir::model::common {

  /**
   * Enum representing available performance models.
   */
  enum class Model {
    ZEN5,
    M4,
    A76,
    GENERIC
  };

  /**
   * Enum representing data types.
   */
  enum class DType {
    FP32,
    FP64
  };

  /**
   * Get the estimated execution time using a performance model.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_trans_a The transpose flag for matrix A (0 or 1).
   * @param i_trans_b The transpose flag for matrix B (0 or 1).
   * @param i_dtype The data type (FP32 or FP64).
   * @param i_model The performance model to use.
   * @param i_peak_gflops Optional peak GFLOPS for generic model (default: 0.0).
   * @param i_vector_size Optional vector width for generic model (default: 0).
   * @param o_gflops Output parameter to receive the estimated GFLOPS.
   *
   * @return The estimated execution time in seconds.
   */
  double get_time_model(int i_m,
                        int i_n,
                        int i_k,
                        int i_trans_a,
                        int i_trans_b,
                        DType i_dtype,
                        Model i_model,
                        double& o_gflops,
                        double i_peak_gflops = 0.0,
                        int i_vector_size = 0);

}  // namespace einsum_ir::model::common

#endif  // EINSUM_IR_MODEL_COMMON_COMMON_H
