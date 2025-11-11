#include "ATen/ATen.h"
#include "catch.hpp"
#include "ContractionBackendBlas.h"


TEST_CASE( "Simple FP32 matmul using the BLAS contraction backend implementation.", "[contraction_backend_blas]" ) {
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

  using namespace einsum_ir::basic;

  at::Tensor l_left    = at::randn( { 8, 5 } );
  at::Tensor l_right   = at::randn( { 7, 8 } );
  at::Tensor l_out     = at::randn( { 7, 5 } );
  at::Tensor l_out_ref = l_out.clone();

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                  m, n, k
  std::vector< int64_t > l_loop_sizes            = {  5, 7, 8 };  
  std::vector< int64_t > l_loop_strides_left     = {  1, 0, 5 };
  std::vector< int64_t > l_loop_strides_right    = {  0, 8, 1 };
  std::vector< int64_t > l_loop_strides_out_aux  = {  0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out      = {  1, 5, 0 };
  std::vector< int64_t > l_packing_strides_left  = {};
  std::vector< int64_t > l_packing_strides_right = {};


  ContractionBackendBlas l_cont_blas;

  l_cont_blas.init( l_loop_dim_type,
                    l_loop_exec_type,
                    l_loop_sizes,
                    l_loop_strides_left,
                    l_loop_strides_right,
                    l_loop_strides_out_aux,
                    l_loop_strides_out,
                    l_packing_strides_left,
                    l_packing_strides_right,
                    data_t::FP32,
                    data_t::FP32,
                    data_t::FP32,
                    data_t::FP32,
                    kernel_t::ZERO,
                    kernel_t::MADD,
                    kernel_t::UNDEFINED_KTYPE,
                    2,
                    2,
                    2,
                    nullptr );                        
  err_t l_err = l_cont_blas.compile();
  REQUIRE(l_err == err_t::SUCCESS );

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref = at::matmul( l_right, l_left );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Simple FP64 matmul using the BLAS contraction backend implementation.", "[contraction_backend_blas]" ) {
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
 
  using namespace einsum_ir::basic;

  at::Tensor l_left    = at::randn( { 8, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_right   = at::randn( { 7, 8 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out     = at::randn( { 7, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out_ref = l_out.clone();

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                  m, n, k
  std::vector< int64_t > l_loop_sizes            = {  5, 7, 8 };  
  std::vector< int64_t > l_loop_strides_left     = {  1, 0, 5 };
  std::vector< int64_t > l_loop_strides_right    = {  0, 8, 1 };
  std::vector< int64_t > l_loop_strides_out_aux  = {  0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out      = {  1, 5, 0 };
  std::vector< int64_t > l_packing_strides_left  = {};
  std::vector< int64_t > l_packing_strides_right = {};


  ContractionBackendBlas l_cont_blas;

  l_cont_blas.init( l_loop_dim_type,
                    l_loop_exec_type,
                    l_loop_sizes,
                    l_loop_strides_left,
                    l_loop_strides_right,
                    l_loop_strides_out_aux,
                    l_loop_strides_out,
                    l_packing_strides_left,
                    l_packing_strides_right,
                    data_t::FP64,
                    data_t::FP64,
                    data_t::FP64,
                    data_t::FP64,
                    kernel_t::ZERO,
                    kernel_t::MADD,
                    kernel_t::UNDEFINED_KTYPE,
                    3,
                    2,
                    4,
                    nullptr );                        
  err_t l_err = l_cont_blas.compile();
  REQUIRE(l_err == err_t::SUCCESS );

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref = at::matmul( l_right, l_left );

  REQUIRE( at::allclose( l_out, l_out_ref ) );
}


TEST_CASE( "Simple batched FP64 matmul using the BLAS contraction backend implementation.", "[contraction_backend_blas]" ) {
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

  using namespace einsum_ir::basic;

  at::Tensor l_left    = at::randn( { 3, 8, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_right   = at::randn( { 3, 7, 8 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out     = at::randn( { 3, 7, 5 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out_ref = l_out.clone();

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C, 
                                             dim_t::M,
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ, 
                                             exec_t::PRIM,
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                  c, m, n, k
  std::vector< int64_t > l_loop_sizes            = {  3, 5, 7, 8 };  
  std::vector< int64_t > l_loop_strides_left     = { 40, 1, 0, 5 };
  std::vector< int64_t > l_loop_strides_right    = { 56, 0, 8, 1 };
  std::vector< int64_t > l_loop_strides_out_aux  = {  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out      = { 35, 1, 5, 0 };
  std::vector< int64_t > l_packing_strides_left  = {};
  std::vector< int64_t > l_packing_strides_right = {};


  ContractionBackendBlas l_cont_blas;

  l_cont_blas.init( l_loop_dim_type,
                    l_loop_exec_type,
                    l_loop_sizes,
                    l_loop_strides_left,
                    l_loop_strides_right,
                    l_loop_strides_out_aux,
                    l_loop_strides_out,
                    l_packing_strides_left,
                    l_packing_strides_right,
                    data_t::FP64,
                    data_t::FP64,
                    data_t::FP64,
                    data_t::FP64,
                    kernel_t::UNDEFINED_KTYPE,
                    kernel_t::MADD,
                    kernel_t::UNDEFINED_KTYPE,
                    4,
                    1,
                    1,
                    nullptr );                        
  err_t l_err = l_cont_blas.compile();
  REQUIRE(l_err == err_t::SUCCESS );

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref += at::matmul( l_right, l_left );

  REQUIRE( at::allclose( l_out, l_out_ref ) );
}

TEST_CASE( "FP32 tensor contraction using the BLAS contraction backend implementation.", "[contraction_backend_blas]" ) {
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

  using namespace einsum_ir::basic;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C, 
                                             dim_t::M,
                                             dim_t::N, 
                                             dim_t::K,
                                             dim_t::M,
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::PRIM,
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                   a,  b,  c,  d, g, f, e
  std::vector< int64_t > l_loop_sizes            = {   5,  2,  8,  3,12, 7, 9 };  
  std::vector< int64_t > l_loop_strides_left     = { 648,324,  0,108, 1, 0,12 };
  std::vector< int64_t > l_loop_strides_right    = {1512,  0,189, 63, 0, 9, 1 };
  std::vector< int64_t > l_loop_strides_out_aux  = {   0,  0,  0,  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out      = {1344,672, 84,  0, 1,12, 0 };
  std::vector< int64_t > l_packing_strides_left  = {};
  std::vector< int64_t > l_packing_strides_right = {};


  ContractionBackendBlas l_cont_blas;

  l_cont_blas.init( l_loop_dim_type,
                    l_loop_exec_type,
                    l_loop_sizes,
                    l_loop_strides_left,
                    l_loop_strides_right,
                    l_loop_strides_out_aux,
                    l_loop_strides_out,
                    l_packing_strides_left,
                    l_packing_strides_right,
                    data_t::FP32,
                    data_t::FP32,
                    data_t::FP32,
                    data_t::FP32,
                    kernel_t::ZERO,
                    kernel_t::MADD,
                    kernel_t::UNDEFINED_KTYPE,
                    5,
                    4,
                    2,
                    nullptr );                        
  err_t l_err = l_cont_blas.compile();
  REQUIRE(l_err == err_t::SUCCESS );

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref = at::einsum( "abdeg,acdfe->abcfg",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Simple packed FP64 matmul using the BLAS contraction backend implementation.", "[contraction_backend_blas]" ) {
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

  using namespace einsum_ir::basic;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C, 
                                             dim_t::M,
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::PRIM, 
                                             exec_t::PRIM,
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                  c, m, n, k
  std::vector< int64_t > l_loop_sizes            = {  3, 5, 7, 8 };  
  std::vector< int64_t > l_loop_strides_left     = { 40, 1, 0, 5 };
  std::vector< int64_t > l_loop_strides_right    = { 56, 0, 8, 1 };
  std::vector< int64_t > l_loop_strides_out_aux  = {  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out      = {  1, 3,15, 0 };
  std::vector< int64_t > l_packing_strides_left  = {};
  std::vector< int64_t > l_packing_strides_right = {};


  ContractionBackendBlas l_cont_blas;

  l_cont_blas.init( l_loop_dim_type,
                    l_loop_exec_type,
                    l_loop_sizes,
                    l_loop_strides_left,
                    l_loop_strides_right,
                    l_loop_strides_out_aux,
                    l_loop_strides_out,
                    l_packing_strides_left,
                    l_packing_strides_right,
                    data_t::FP64,
                    data_t::FP64,
                    data_t::FP64,
                    data_t::FP64,
                    kernel_t::COPY,
                    kernel_t::PACKED_MADD,
                    kernel_t::COPY,
                    6,
                    3,
                    5,
                    nullptr );                        
  err_t l_err = l_cont_blas.compile();
  REQUIRE(l_err == err_t::SUCCESS );

  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref += at::einsum( "ckm,cnk->nmc",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref ) );
}

TEST_CASE( "FP32 packed tensor contraction using the BLAS contraction backend implementation.", "[contraction_backend_blas]" ) {
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
  //    h      4  C (packed GEMMM)
  //                                  a  b  d  h  e   g
  at::Tensor l_left    = at::randn( { 5, 2, 3, 4, 9, 12 } );
  //                                  a  c  d  h  f   e
  at::Tensor l_right   = at::randn( { 5, 8, 3, 4, 7,  9 } );
  //                                  a  b  c  f   g  h
  at::Tensor l_out     = at::randn( { 5, 2, 8, 7, 12, 4 } );
  at::Tensor l_out_ref = l_out.clone();

  using namespace einsum_ir::basic;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C, 
                                             dim_t::M,
                                             dim_t::N, 
                                             dim_t::K,
                                             dim_t::C, 
                                             dim_t::M,
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::PRIM,
                                             exec_t::PRIM,
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                   a,   b,  c,  d,  h, g, f, e
  std::vector< int64_t > l_loop_sizes            = {   5,   2,  8,  3,  4,12, 7, 9 };  
  std::vector< int64_t > l_loop_strides_left     = {2592,1296,  0,432,108, 1, 0,12 };
  std::vector< int64_t > l_loop_strides_right    = {6048,   0,756,252, 63, 0, 9, 1 };
  std::vector< int64_t > l_loop_strides_out_aux  = {   0,   0,  0,  0,  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out      = {5376,2688,336,  0,  1, 4,48, 0 };
  std::vector< int64_t > l_packing_strides_left  = {};
  std::vector< int64_t > l_packing_strides_right = {};


  ContractionBackendBlas l_cont_blas;

  l_cont_blas.init( l_loop_dim_type,
                    l_loop_exec_type,
                    l_loop_sizes,
                    l_loop_strides_left,
                    l_loop_strides_right,
                    l_loop_strides_out_aux,
                    l_loop_strides_out,
                    l_packing_strides_left,
                    l_packing_strides_right,
                    data_t::FP32,
                    data_t::FP32,
                    data_t::FP32,
                    data_t::FP32,
                    kernel_t::ZERO,
                    kernel_t::PACKED_MADD,
                    kernel_t::COPY,
                    7,
                    5,
                    1,
                    nullptr );                        
  err_t l_err = l_cont_blas.compile();
  REQUIRE(l_err == err_t::SUCCESS );
  
  l_cont_blas.contract( l_left.data_ptr(),
                        l_right.data_ptr(),
                        nullptr,
                        l_out.data_ptr() );

  l_out_ref = at::einsum( "abdheg,acdhfe->abcfgh",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}