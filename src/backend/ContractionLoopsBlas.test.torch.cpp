#include "ATen/ATen.h"
#include "catch.hpp"
#include "ContractionLoopsBlas.h"
#include <cmath>

TEST_CASE( "Simple FP32 matmul using the BLAS contraction loops implementation.", "[contraction_loops_blas]" ) {
  // test case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      5
  //    n    1      7
  //    k    2      8

  at::Tensor l_left    = at::randn( { 8, 5 } );
  at::Tensor l_right   = at::randn( { 7, 8 } );
  at::Tensor l_out     = at::randn( { 7, 5 } );
  at::Tensor l_out_ref = l_out.clone();

  einsum_ir::backend::ContractionLoopsBlas l_cont_blas;

  l_cont_blas.init( 0,
                    0,
                    0,
                    0,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    einsum_ir::FP32,
                    false,
                    false,
                    false,
                    1,
                    5,
                    7,
                    8,
                    5,
                    8,
                    5,
                    1.0,
                    1.0,
                    einsum_ir::kernel_t::ZERO,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE );                    
  l_cont_blas.compile();

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref = at::matmul( l_right, l_left );

  REQUIRE( at::allclose( l_out, l_out_ref ) );
}

TEST_CASE( "Simple FP64 matmul using the BLAS contraction loops implementation.", "[contraction_loops_blas]" ) {
  // test case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      5
  //    n    1      7
  //    k    2      8

  at::Tensor l_left    = at::randn( { 8, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_right   = at::randn( { 7, 8 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out     = at::randn( { 7, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out_ref = l_out.clone();

  einsum_ir::backend::ContractionLoopsBlas l_cont_blas;

  l_cont_blas.init( 0,
                    0,
                    0,
                    0,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    einsum_ir::FP64,
                    false,
                    false,
                    false,
                    1,
                    5,
                    7,
                    8,
                    5,
                    8,
                    5,
                    1.0,
                    1.0,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE );
  l_cont_blas.compile();

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref += at::matmul( l_right, l_left );

  REQUIRE( at::allclose( l_out, l_out_ref ) );
}

TEST_CASE( "Simple batched FP64 matmul using the BLAS contraction loops implementation.", "[contraction_loops_blas]" ) {
  // test case:
  //
  //    ____cnm___
  //   /          \
  // ckm           cnk
  //
  // char    size
  //    c    3
  //    m    5
  //    n    7
  //    k    8

  at::Tensor l_left    = at::randn( { 3, 8, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_right   = at::randn( { 3, 7, 8 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out     = at::randn( { 3, 7, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out_ref = l_out.clone();

  einsum_ir::backend::ContractionLoopsBlas l_cont_blas;

  int64_t l_sizes_c[1] = { 3 };
  int64_t l_strides_in_left_c[1]  = { 8*5 };
  int64_t l_strides_in_right_c[1] = { 7*8 };
  int64_t l_strides_out_aux_c[1] = { 0 };
  int64_t l_strides_out_c[1] = { 7*5 };

  l_cont_blas.init( 1,
                    0,
                    0,
                    0,
                    l_sizes_c,
                    nullptr,
                    nullptr,
                    nullptr,
                    l_strides_in_left_c,
                    nullptr,
                    nullptr,
                    l_strides_in_right_c,
                    nullptr,
                    nullptr,
                    l_strides_out_aux_c,
                    nullptr,
                    nullptr,
                    l_strides_out_c,
                    nullptr,
                    nullptr,
                    einsum_ir::FP64,
                    false,
                    false,
                    false,
                    1,
                    5,
                    7,
                    8,
                    5,
                    8,
                    5,
                    1.0,
                    1.0,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE );
  l_cont_blas.compile();

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref += at::matmul( l_right, l_left );

  REQUIRE( at::allclose( l_out, l_out_ref ) );
}

TEST_CASE( "FP32 tensor contraction using the BLAS contraction loops implementation.", "[contraction_loops_blas]" ) {
  // test case:
  //
  //    ____abcfg___
  //   /            \
  // abdeg         acdfe
  //
  // char   size  dimension type
  //    a      5  C
  //    b      2  M
  //    c      8  N
  //    d      3  K
  //    e      9  K (BLAS)
  //    g     12  M (BLAS)
  //    f      7  N (BLAS)
  //                                  a  b  d  e   g
  at::Tensor l_left    = at::randn( { 5, 2, 3, 9, 12 } );
  //                                  a  c  d  f   e
  at::Tensor l_right   = at::randn( { 5, 8, 3, 7,  9 } );
  //                                  a  b  c  f   g
  at::Tensor l_out     = at::randn( { 5, 2, 8, 7, 12 } );
  at::Tensor l_out_ref = l_out.clone();

  int64_t l_sizes_c[1] = { 5 };
  int64_t l_sizes_m[1] = { 2 };
  int64_t l_sizes_n[1] = { 8 };
  int64_t l_sizes_k[1] = { 3 };

  int64_t l_strides_in_left_c[1]  = { 2*3*9*12 };
  int64_t l_strides_in_left_m[1]  = {   3*9*12 };
  int64_t l_strides_in_left_k[1]  = {     9*12 };

  int64_t l_strides_in_right_c[1] = { 8*3*7*9 };
  int64_t l_strides_in_right_n[1] = {   3*7*9 };
  int64_t l_strides_in_right_k[1] = {     7*9 };

  int64_t l_strides_out_aux_c[1] = { 0 };
  int64_t l_strides_out_aux_m[1] = { 0 };
  int64_t l_strides_out_aux_n[1] = { 0 };

  int64_t l_strides_out_c[1] = { 2*8*7*12 };
  int64_t l_strides_out_m[1] = {   8*7*12 };
  int64_t l_strides_out_n[1] = {     7*12 };

  einsum_ir::backend::ContractionLoopsBlas l_cont_blas;

  l_cont_blas.init( 1,
                    1,
                    1,
                    1,
                    l_sizes_c,
                    l_sizes_m,
                    l_sizes_n,
                    l_sizes_k,
                    l_strides_in_left_c,
                    l_strides_in_left_m,
                    l_strides_in_left_k,
                    l_strides_in_right_c,
                    l_strides_in_right_n,
                    l_strides_in_right_k,
                    l_strides_out_aux_c,
                    l_strides_out_aux_m,
                    l_strides_out_aux_n,
                    l_strides_out_c,
                    l_strides_out_m,
                    l_strides_out_n,
                    einsum_ir::FP32,
                    false,
                    false,
                    false,
                    1,
                    12,
                    7,
                    9,
                    12,
                    9,
                    12,
                    1.0,
                    1.0,
                    einsum_ir::kernel_t::ZERO,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE );

  l_cont_blas.compile();

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref = at::einsum( "abdeg,acdfe->abcfg",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Simple packed FP64 matmul using the BLAS contraction loops implementation.", "[contraction_loops_blas]" ) {
  // test case:
  //
  //    ____nmc___
  //   /          \
  // ckm           cnk
  //
  // char    size
  //    c    3
  //    m    5
  //    n    7
  //    k    8

  at::Tensor l_left    = at::randn( { 3, 8, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_right   = at::randn( { 3, 7, 8 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out     = at::randn( { 7, 5, 3 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out_ref = l_out.clone();

  einsum_ir::backend::ContractionLoopsBlas l_cont_blas;

  l_cont_blas.init( 0,
                    0,
                    0,
                    0,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    einsum_ir::FP64,
                    false,
                    false,
                    false,
                    3,
                    5,
                    7,
                    8,
                    5,
                    8,
                    15,
                    1.0,
                    1.0,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE );
  l_cont_blas.compile();

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref += at::einsum( "ckm,cnk->nmc",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref ) );
}

TEST_CASE( "FP32 packed tensor contraction using the BLAS contraction loops implementation.", "[contraction_loops_blas]" ) {
  // test case:
  //
  //    ____abcfgh___
  //   /             \
  // abdheg         acdhfe
  //
  // char   size  dimension type
  //    a      5  C
  //    b      2  M
  //    c      8  N
  //    d      3  K
  //    e      9  K (BLAS)
  //    f      7  N (BLAS)
  //    g     12  M (BLAS)
  //    h      9  C (packed GEMMM)
  //                                  a  b  d  h  e   g
  at::Tensor l_left    = at::randn( { 5, 2, 3, 4, 9, 12 } );
  //                                  a  c  d  h  f   e
  at::Tensor l_right   = at::randn( { 5, 8, 3, 4, 7,  9 } );
  //                                  a  b  c  f   g  h
  at::Tensor l_out     = at::randn( { 5, 2, 8, 7, 12, 4 } );
  at::Tensor l_out_ref = l_out.clone();

  int64_t l_sizes_c[1] = { 5 };
  int64_t l_sizes_m[1] = { 2 };
  int64_t l_sizes_n[1] = { 8 };
  int64_t l_sizes_k[1] = { 3 };

  int64_t l_strides_in_left_c[1]  = { 2*3*4*9*12 };
  int64_t l_strides_in_left_m[1]  = {   3*4*9*12 };
  int64_t l_strides_in_left_k[1]  = {     4*9*12 };

  int64_t l_strides_in_right_c[1] = { 8*3*4*7*9 };
  int64_t l_strides_in_right_n[1] = {   3*4*7*9 };
  int64_t l_strides_in_right_k[1] = {     4*7*9 };

  int64_t l_strides_out_aux_c[1] = { 0 };
  int64_t l_strides_out_aux_m[1] = { 0 };
  int64_t l_strides_out_aux_n[1] = { 0 };

  int64_t l_strides_out_c[1] = { 2*8*7*12*4 };
  int64_t l_strides_out_m[1] = {   8*7*12*4 };
  int64_t l_strides_out_n[1] = {     7*12*4 };

  einsum_ir::backend::ContractionLoopsBlas l_cont_blas;

  l_cont_blas.init( 1,
                    1,
                    1,
                    1,
                    l_sizes_c,
                    l_sizes_m,
                    l_sizes_n,
                    l_sizes_k,
                    l_strides_in_left_c,
                    l_strides_in_left_m,
                    l_strides_in_left_k,
                    l_strides_in_right_c,
                    l_strides_in_right_n,
                    l_strides_in_right_k,
                    l_strides_out_aux_c,
                    l_strides_out_aux_m,
                    l_strides_out_aux_n,
                    l_strides_out_c,
                    l_strides_out_m,
                    l_strides_out_n,
                    einsum_ir::FP32,
                    false,
                    false,
                    false,
                    4,
                    12,
                    7,
                    9,
                    12,
                    9,
                    4*12,
                    1.0,
                    1.0,
                    einsum_ir::kernel_t::ZERO,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE );

  l_cont_blas.compile();

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref = at::einsum( "abdheg,acdhfe->abcfgh",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}