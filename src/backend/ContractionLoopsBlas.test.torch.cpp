#include "ATen/ATen.h"
#include "catch.hpp"
#include "ContractionLoopsBlas.h"
#include "../constants.h"
#include <cmath>
#include <map>
#include <vector>


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
                    einsum_ir::FP32,
                    false,
                    false,
                    1,
                    5,
                    7,
                    8,
                    5,
                    8,
                    5,
                    einsum_ir::kernel_t::ZERO,
                    einsum_ir::kernel_t::MADD,
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
                    einsum_ir::FP64,
                    false,
                    false,
                    1,
                    5,
                    7,
                    8,
                    5,
                    8,
                    5,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE,
                    einsum_ir::kernel_t::MADD,
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

  int64_t l_id_c = 0;

  std::vector<int64_t> l_dim_ids_c = { l_id_c };

  // per-dimension sizes
  std::map< int64_t, int64_t > l_dim_sizes{ { l_id_c, 3 } };
  
  std::map< int64_t, einsum_ir::dim_t > l_dim_types{ { l_id_c, einsum_ir::C } };

  std::map< int64_t, int64_t > l_strides_in_left{  { l_id_c, 8*5 } };
  std::map< int64_t, int64_t > l_strides_in_right{ { l_id_c, 7*8 } };
  std::map< int64_t, int64_t > l_strides_out_aux{  { l_id_c, 0   } };
  std::map< int64_t, int64_t > l_strides_out{      { l_id_c, 7*5 } };

  l_cont_blas.init( l_dim_ids_c.size(),
                    0,
                    0,
                    0,
                    l_dim_ids_c.data(),
                    nullptr,
                    nullptr,
                    nullptr,
                    &l_dim_sizes,
                    &l_strides_in_left,
                    &l_strides_in_right,
                    &l_strides_out_aux,
                    &l_strides_out,
                    &l_dim_types,
                    einsum_ir::FP64,
                    false,
                    false,
                    1,
                    5,
                    7,
                    8,
                    5,
                    8,
                    5,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE,
                    einsum_ir::kernel_t::MADD,
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

  int64_t l_id_c = 0;
  int64_t l_id_m = 1;
  int64_t l_id_n = 2;
  int64_t l_id_k = 3;

  std::vector<int64_t> l_dim_ids_c = { l_id_c };
  std::vector<int64_t> l_dim_ids_m = { l_id_m };
  std::vector<int64_t> l_dim_ids_n = { l_id_n };
  std::vector<int64_t> l_dim_ids_k = { l_id_k };

  std::map< int64_t, int64_t > l_dim_sizes{ { l_id_c, 5 },
                                            { l_id_m, 2 },
                                            { l_id_n, 8 },
                                            { l_id_k, 3 } };
  
  std::map< int64_t, einsum_ir::dim_t > l_dim_types{ { l_id_c, einsum_ir::C },
                                                     { l_id_m, einsum_ir::M },
                                                     { l_id_n, einsum_ir::N },
                                                     { l_id_k, einsum_ir::K } };


  std::map< int64_t, int64_t > l_strides_in_left{ { l_id_c, 2*3*9*12 },
                                                  { l_id_m,   3*9*12 },
                                                  { l_id_k,     9*12 } };

  std::map< int64_t, int64_t > l_strides_in_right{ { l_id_c, 8*3*7*9 },
                                                   { l_id_n,   3*7*9 },
                                                   { l_id_k,     7*9 } };

  std::map< int64_t, int64_t > l_strides_out_aux{ };

  std::map< int64_t, int64_t > l_strides_out{ { l_id_c, 2*8*7*12 },
                                              { l_id_m,   8*7*12 },
                                              { l_id_n,     7*12 } };

  einsum_ir::backend::ContractionLoopsBlas l_cont_blas;

  l_cont_blas.init( l_dim_ids_c.size(),
                    l_dim_ids_m.size(),
                    l_dim_ids_n.size(),
                    l_dim_ids_k.size(),
                    l_dim_ids_c.data(),
                    l_dim_ids_m.data(),
                    l_dim_ids_n.data(),
                    l_dim_ids_k.data(),
                    &l_dim_sizes,
                    &l_strides_in_left,
                    &l_strides_in_right,
                    &l_strides_out_aux,
                    &l_strides_out,
                    &l_dim_types,
                    einsum_ir::FP32,
                    false,
                    false,
                    1,
                    12,
                    7,
                    9,
                    12,
                    9,
                    12,
                    einsum_ir::kernel_t::ZERO,
                    einsum_ir::kernel_t::MADD,
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
                    einsum_ir::FP64,
                    false,
                    false,
                    3,
                    5,
                    7,
                    8,
                    5,
                    8,
                    15,
                    einsum_ir::kernel_t::UNDEFINED_KTYPE,
                    einsum_ir::kernel_t::MADD,
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

  int64_t l_id_c = 0;
  int64_t l_id_m = 1;
  int64_t l_id_n = 2;
  int64_t l_id_k = 3;

  std::vector<int64_t> l_dim_ids_c = { l_id_c };
  std::vector<int64_t> l_dim_ids_m = { l_id_m };
  std::vector<int64_t> l_dim_ids_n = { l_id_n };
  std::vector<int64_t> l_dim_ids_k = { l_id_k };

  std::map< int64_t, int64_t > l_dim_sizes{ { l_id_c, 5 },
                                            { l_id_m, 2 },
                                            { l_id_n, 8 },
                                            { l_id_k, 3 } };
  
  std::map< int64_t, einsum_ir::dim_t > l_dim_types{ { l_id_c, einsum_ir::C },
                                                     { l_id_m, einsum_ir::M },
                                                     { l_id_n, einsum_ir::N },
                                                     { l_id_k, einsum_ir::K } };


  std::map< int64_t, int64_t > l_strides_in_left{ { l_id_c, 2*3*4*9*12 },
                                                  { l_id_m,   3*4*9*12 },
                                                  { l_id_k,     4*9*12 } };

  std::map< int64_t, int64_t > l_strides_in_right{ { l_id_c, 8*3*4*7*9 },
                                                   { l_id_n,   3*4*7*9 },
                                                   { l_id_k,     4*7*9 } };

  std::map< int64_t, int64_t > l_strides_out_aux{ };

  std::map< int64_t, int64_t > l_strides_out{ { l_id_c, 2*8*7*12*4 },
                                              { l_id_m,   8*7*12*4 },
                                              { l_id_n,     7*12*4 } };

  einsum_ir::backend::ContractionLoopsBlas l_cont_blas;

  l_cont_blas.init( l_dim_ids_c.size(),
                    l_dim_ids_m.size(),
                    l_dim_ids_n.size(),
                    l_dim_ids_k.size(),
                    l_dim_ids_c.data(),
                    l_dim_ids_m.data(),
                    l_dim_ids_n.data(),
                    l_dim_ids_k.data(),
                    &l_dim_sizes,
                    &l_strides_in_left,
                    &l_strides_in_right,
                    &l_strides_out_aux,
                    &l_strides_out,
                    &l_dim_types,
                    einsum_ir::FP32,
                    false,
                    false,
                    4,
                    12,
                    7,
                    9,
                    12,
                    9,
                    4*12,
                    einsum_ir::kernel_t::ZERO,
                    einsum_ir::kernel_t::MADD,
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