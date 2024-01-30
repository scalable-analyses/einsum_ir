#include <ATen/ATen.h>
#include "catch.hpp"
#include "EinsumExpression.h"

TEST_CASE( "Single matmul example using an einsum expression through the native interface.", "[einsum_exp]" ) {
  // test case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4

  // data
  at::Tensor l_left    = at::rand( {4, 2} );
  at::Tensor l_right   = at::rand( {3, 4} );
  at::Tensor l_out_ref = at::rand( {3, 2} );
  at::Tensor l_out     = l_out_ref.clone();

  int64_t l_dim_sizes[3] = { 2, 3, 4 };

  int64_t l_string_dim_ids[6] = { 2, 0,   // km
                                  1, 2,   // nk
                                  1, 0 }; // nm

  int64_t l_string_num_dims[3] = { 2, 2, 2 };

  void * l_data_ptrs[3] = { l_left.data_ptr(),
                            l_right.data_ptr(),
                            l_out.data_ptr() };

  int64_t l_path[2] = { 0, 1 };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;

  l_einsum_exp.init( 3,
                     l_dim_sizes,
                     1,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  l_out_ref = at::einsum( "km,nk->nm",
                          {l_left, l_right} );

  // check results
  REQUIRE( at::allclose( l_out, l_out_ref )  );

  // check number of flops
  REQUIRE( l_einsum_exp.num_ops() == 2*3*4*2 - 2*3 );
}

TEST_CASE( "Single batch-outer complex matmul example using an einsum expression through the native interface.", "[einsum_exp]" ) {
  // test case:
  //
  //    ____cnm___
  //   /          \
  // ckm          cnk
  //
  // char   id   size
  //    c    0      2
  //    m    1      2
  //    n    2      3
  //    k    3      4

  // data
  at::Tensor l_data_ckm = at::randn( {2, 4, 2} );
  at::Tensor l_data_cnk = at::randn( {2, 3, 4} );
  at::Tensor l_data_cnm = at::randn( {2, 3, 2} );

  int64_t l_dim_sizes[4] = { 2, 2, 3, 4 };

  int64_t l_string_dim_ids[9] = { 0, 3, 1,   // ckm
                                  0, 2, 3,   // cnk
                                  0, 2, 1 }; // cnm

  int64_t l_string_num_dims[3] = { 3, 3, 3 };

  void * l_data_ptrs[3] = { l_data_ckm.data_ptr(),
                            l_data_cnk.data_ptr(),
                            l_data_cnm.data_ptr() };

  int64_t l_path[2] = { 0, 1 }; 

  einsum_ir::frontend::EinsumExpression l_einsum_exp;

  l_einsum_exp.init( 4,
                     l_dim_sizes,
                     1,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::complex_t::BATCH_OUTER,
                     einsum_ir::data_t::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  at::Tensor l_data_kmc = at::view_as_complex( l_data_ckm.permute( {1, 2, 0} ).contiguous() );
  at::Tensor l_data_nkc = at::view_as_complex( l_data_cnk.permute( {1, 2, 0} ).contiguous() );
  at::Tensor l_data_nmc_ref = at::einsum( "km,nk->nm",
                                          {l_data_kmc, l_data_nkc} );

  // check results
  at::Tensor l_data_nmc = at::view_as_complex( l_data_cnm.permute( {1, 2, 0} ).contiguous() );

  REQUIRE( at::allclose( l_data_nmc, l_data_nmc_ref )  );

  // check number of flops
  REQUIRE( l_einsum_exp.num_ops() == 4 * 2*3*4*2 - 2 * 2*3 );
}

TEST_CASE( "Single batch-inner complex matmul example using an einsum expression through the native interface.", "[einsum_exp]" ) {
  // test case:
  //
  //    ___nmc___
  //   /         \
  // kmc         nkc
  //
  // char   id   size
  //    c    0      2
  //    m    1      2
  //    n    2      3
  //    k    3      4

  // data
  at::Tensor l_data_kmc = at::randn( {4, 2},
                                     at::ScalarType::ComplexFloat );
  at::Tensor l_data_nkc = at::randn( {3, 4},
                                     at::ScalarType::ComplexFloat );
  at::Tensor l_data_nmc = at::randn( {3, 2},
                                     at::ScalarType::ComplexFloat );

  int64_t l_dim_sizes[4] = { 2, 2, 3, 4 };

  int64_t l_string_dim_ids[9] = { 3, 1, 0,   // kmc
                                  2, 3, 0,   // nkc
                                  2, 1, 0 }; // nmc

  int64_t l_string_num_dims[3] = { 3, 3, 3 };

  void * l_data_ptrs[3] = { l_data_kmc.data_ptr(),
                            l_data_nkc.data_ptr(),
                            l_data_nmc.data_ptr() };

  int64_t l_path[2] = { 0, 1 }; 

  einsum_ir::frontend::EinsumExpression l_einsum_exp;

  l_einsum_exp.init( 4,
                     l_dim_sizes,
                     1,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::complex_t::BATCH_INNER,
                     einsum_ir::data_t::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  at::Tensor l_data_nmc_ref = at::einsum( "km,nk->nm",
                                          {l_data_kmc, l_data_nkc} );

  // check results
  REQUIRE( at::allclose( l_data_nmc, l_data_nmc_ref )  );

  // check number of flops
  REQUIRE( l_einsum_exp.num_ops() == 4 * 2*3*4*2 - 2 * 2*3 );
}

TEST_CASE( "Binary contraction representing a sum of small GEMMs.", "[einsum_exp]" ) {
  // test case:
  //
  //       __bd__
  //      /      \
  //   cad       cba
  //
  // char   id   size
  //    a    0     96
  //    b    1     24
  //    c    2   4096
  //    d    3     96

  // data
  at::Tensor l_left    = at::randn( {4096, 96, 96} ) / 4096;
  at::Tensor l_right   = at::randn( {4096, 24, 96} ) / 4096;
  at::Tensor l_out_ref = at::randn( {24, 96} );
  at::Tensor l_out     = l_out_ref.clone();

  int64_t l_dim_sizes[5] = { 96, 24, 4096, 96 };

  int64_t l_string_dim_ids[10] = { 2, 0, 3, // cad
                                   2, 1, 0, // cba
                                   1, 3 };  // bd

  int64_t l_string_num_dims[3] = { 3, 3, 2 };

  void * l_data_ptrs[3] = { l_left.data_ptr(),
                            l_right.data_ptr(),
                            l_out.data_ptr() };

  int64_t l_path[2] = { 0, 1 };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;

  l_einsum_exp.init( 4,
                     l_dim_sizes,
                     1,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  l_out_ref = at::einsum( "cad,cba->bd",
                          {l_left, l_right} );

  // check results
  REQUIRE( at::allclose( l_out,
                         l_out_ref )  );
}

TEST_CASE( "Binary contraction representing packed GEMM.", "[einsum_exp_tmp]" ) {
  // test case:
  //
  //       __bdc__
  //      /       \
  //    adc       bac
  //
  // char   id   size
  //    a    0     32
  //    b    1     24
  //    c    2     48
  //    d    3     16

  // data
  at::Tensor l_data_adc     = at::rand( {32, 16, 48 } );
  at::Tensor l_data_bac     = at::rand( {24, 32, 48 } );
  at::Tensor l_data_bdc_ref = at::rand( {24, 16, 48 } );
  at::Tensor l_data_bdc     = l_data_bdc_ref.clone();

  int64_t l_dim_sizes[4] = { 32, 24, 48, 16 };

  int64_t l_string_dim_ids[9] = { 0, 3, 2,   // adc
                                  1, 0, 2,    // bac
                                  1, 3, 2 };  // bdc

  int64_t l_string_num_dims[3] = { 3, 3, 3 };

  void * l_data_ptrs[3] = { l_data_adc.data_ptr(),
                            l_data_bac.data_ptr(),
                            l_data_bdc.data_ptr() };

  int64_t l_path[2] = { 0, 1 };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;

  l_einsum_exp.init( 4,
                     l_dim_sizes,
                     1,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  l_data_bdc_ref = at::einsum( "adc,bac->bdc",
                               {l_data_adc, l_data_bac} );

  // check results
  REQUIRE( at::allclose( l_data_bdc_ref,
                         l_data_bdc,
                         1e-5,
                         7e-5 )  );
}

TEST_CASE( "Binary contraction using an einsum expression through the native interface.", "[einsum_exp]" ) {
  // test case:
  //
  //       __be__
  //      /      \
  //   abcd      deca
  //
  // char   id   size
  //    a    0     96
  //    b    1     24
  //    c    2     84
  //    d    3     96
  //    e    4     84

  // data
  at::Tensor l_left    = at::rand( {96, 24, 84, 96},
                                   at::ScalarType::Double );
  at::Tensor l_right   = at::rand( {96, 84, 84, 96},
                                   at::ScalarType::Double );
  at::Tensor l_out_ref = at::rand( {24, 84},
                                   at::ScalarType::Double );
  at::Tensor l_out     = l_out_ref.clone();

  int64_t l_dim_sizes[5] = { 96, 24, 84, 96, 84 };

  int64_t l_string_dim_ids[10] = { 0, 1, 2, 3, // abcd
                                   3, 4, 2, 0, // deca
                                   1, 4 };     // be

  int64_t l_string_num_dims[3] = { 4, 4, 2 };

  void * l_data_ptrs[3] = { l_left.data_ptr(),
                            l_right.data_ptr(),
                            l_out.data_ptr() };

  int64_t l_path[2] = { 0, 1 };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;

  l_einsum_exp.init( 5,
                     l_dim_sizes,
                     1,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP64,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  l_out_ref = at::einsum( "abcd,deca->be",
                          {l_left, l_right} );

  // check results
  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "Two matmul example using an einsum expression through the native interface.", "[einsum_exp]" ) {
  // test case:
  //
  //         __bd__
  //        /      \
  //    ___ba___    da
  //   /        \
  // ca          bc
  //
  // char   id   size
  //    a    0      2
  //    b    1      3
  //    c    2      4
  //    d    3      5
  
  // data
  at::Tensor l_data_ca     = at::rand( {4, 2} );
  at::Tensor l_data_bc     = at::rand( {3, 4} );
  at::Tensor l_data_da     = at::rand( {5, 2} );
  at::Tensor l_data_bd_ref = at::rand( {3, 5} );
  at::Tensor l_data_bd     = l_data_bd_ref.clone();

  int64_t l_dim_sizes[4] = { 2, 3, 4, 5 };

  int64_t l_string_num_dims[4] = { 2, 2, 2, 2 };

  int64_t l_string_dim_ids[8] = { 2, 0,   // ca
                                  1, 2,   // bc
                                  3, 0,   // da
                                  1, 3 }; // bd

  int64_t l_path[4] = { 0, 1,   // ba
                        0, 1 }; // bd 

  void * l_data_ptrs[4] = { l_data_ca.data_ptr(),
                            l_data_bc.data_ptr(),
                            l_data_da.data_ptr(),
                            l_data_bd.data_ptr() };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;

  l_einsum_exp.init( 4,
                     l_dim_sizes,
                     2,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  l_data_bd_ref = at::einsum( "ca,bc,da->bd",
                              {l_data_ca, l_data_bc, l_data_da} );

  // check results
  REQUIRE( at::allclose( l_data_bd, l_data_bd_ref )  );

  // check number of flops
  REQUIRE( l_einsum_exp.num_ops() == 2*3*4*2 - 2*3 + 2*3*5*2 - 3*5 );
}

TEST_CASE( "Two matmul expression with locked data.", "[einsum_exp]" ) {
  // test case:
  //
  //         __bd__
  //        /      \
  //    ___ba___    da
  //   /        \
  // ca          bc
  //
  // char   id   size
  //    a    0      2
  //    b    1      3
  //    c    2      4
  //    d    3      5

  // data
  at::Tensor l_data_ca     = at::rand( {4, 2} );
  at::Tensor l_data_bc     = at::rand( {3, 4} );
  at::Tensor l_data_da     = at::rand( {5, 2} );
  at::Tensor l_data_bd_ref = at::rand( {3, 5} );
  at::Tensor l_data_bd     = l_data_bd_ref.clone();

  int64_t l_dim_sizes[4] = { 2, 3, 4, 5 };

  int64_t l_string_num_dims[4] = { 2, 2, 2, 2 };

  int64_t l_string_dim_ids[8] = { 2, 0,   // ca
                                  1, 2,   // bc
                                  3, 0,   // da
                                  1, 3 }; // bd

  int64_t l_path[4] = { 0, 1,   // ba
                        0, 1 }; // bd

  void * l_data_ptrs[4] = { l_data_ca.data_ptr(),
                            l_data_bc.data_ptr(),
                            l_data_da.data_ptr(),
                            l_data_bd.data_ptr() };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;

  l_einsum_exp.init( 4,
                     l_dim_sizes,
                     2,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // reference
  l_data_bd_ref = at::einsum( "ca,bc,da->bd",
                              {l_data_ca, l_data_bc, l_data_da} );

  // lock input data
  l_err = l_einsum_exp.store_and_lock_data( 0 );
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_einsum_exp.store_and_lock_data( 1 );
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_einsum_exp.store_and_lock_data( 2 );
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // modify input data
  l_data_ca += at::rand( {4, 2} );
  l_data_bc += at::rand( {3, 4} );
  l_data_da += at::rand( {5, 2} );

  l_einsum_exp.eval();

  // check results
  REQUIRE( at::allclose( l_data_bd, l_data_bd_ref )  );


  // unlock input data
  l_err = l_einsum_exp.unlock_data( 0 );
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_einsum_exp.unlock_data( 1 );
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_einsum_exp.unlock_data( 2 );
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  l_data_bd_ref = at::einsum( "ca,bc,da->bd",
                              {l_data_ca, l_data_bc, l_data_da} );

  // check result
  REQUIRE( at::allclose( l_data_bd, l_data_bd_ref )  );
}

TEST_CASE( "Single-level einsum expression using the internal interface, stride-1 N.", "[einsum_exp]" ) {
  // test case:
  //
  // string: fcahy,gcexaiy->xhgfeiy
  // contraction path: [(0, 1)]
  //
  //       _____________yxhgfei______
  //      /                          \
  //   fcahy                       gcexaiy
  //
  // char   id   size
  //    a    0      4
  //    c    1      5
  //    e    2      7
  //    f    3      2
  //    g    4      3
  //    h    5      6
  //    i    6      9
  //    x    7      2
  //    y    8      3
  int64_t l_dim_sizes[9] = { 4, 5, 7, 2, 3, 6, 9, 2, 3 };

  int64_t l_string_num_dims[3] = { 5, 7, 7 };

  int64_t l_string_dim_ids[19] = { 3, 1, 0, 5, 8,         // fcahy
                                   4, 1, 2, 7, 0, 6, 8,   // gcexaiy
                                   8, 7, 5, 4, 3, 2, 6 }; // xhgfeiy

  int64_t l_path[2] = { 0, 1 };

  // data
  at::Tensor l_data_fcahy   = at::randn( { l_dim_sizes[3],
                                           l_dim_sizes[1],
                                           l_dim_sizes[0],
                                           l_dim_sizes[5],
                                           l_dim_sizes[8] },
                                           at::ScalarType::Double );

  at::Tensor l_data_gcexaiy = at::randn( { l_dim_sizes[4],
                                           l_dim_sizes[1],
                                           l_dim_sizes[2],
                                           l_dim_sizes[7],
                                           l_dim_sizes[0],
                                           l_dim_sizes[6],
                                           l_dim_sizes[8] },
                                           at::ScalarType::Double );

  at::Tensor l_data_xhgfeiy = at::randn( { l_dim_sizes[8],
                                           l_dim_sizes[7],
                                           l_dim_sizes[5],
                                           l_dim_sizes[4],
                                           l_dim_sizes[3],
                                           l_dim_sizes[2],
                                           l_dim_sizes[6] },
                                           at::ScalarType::Double );


  void * l_data_ptrs[3] = { l_data_fcahy.data_ptr(),
                            l_data_gcexaiy.data_ptr(),
                            l_data_xhgfeiy.data_ptr() };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;
  l_einsum_exp.init( 9,
                     l_dim_sizes,
                     1,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP64,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  at::Tensor l_data_xhgfeiy_ref = at::einsum( "fcahy,gcexaiy->yxhgfei",
                                              { l_data_fcahy,
                                                l_data_gcexaiy } );

  REQUIRE( at::allclose( l_data_xhgfeiy_ref, l_data_xhgfeiy ) );
}

TEST_CASE( "Single-level einsum expression using the internal interface, stride-1 C.", "[einsum_exp]" ) {
  // test case:
  //
  // string: fcahy,gcexaiy->xhgfeiy
  // contraction path: [(0, 1)] 
  //
  //       _____________xhgfeiy______
  //      /                          \
  //   fcahy                       gcexaiy
  //
  // char   id   size  type (internal)
  //    a    0      4  K
  //    c    1      5  K
  //    e    2      7  M
  //    f    3      2  N
  //    g    4      3  M
  //    h    5      6  N
  //    i    6      9  M
  //    x    7      2  M
  //    y    8      3  C
  int64_t l_dim_sizes[9] = { 4, 5, 7, 2, 3, 6, 9, 2, 3 };

  int64_t l_string_num_dims[3] = { 5, 7, 7 };

  int64_t l_string_dim_ids[19] = { 3, 1, 0, 5, 8,         // fcahy
                                   4, 1, 2, 7, 0, 6, 8,   // gcexaiy
                                   7, 5, 4, 3, 2, 6, 8 }; // xhgfeiy

  int64_t l_path[2] = { 0, 1 };

  // data
  at::Tensor l_data_fcahy   = at::randn( { l_dim_sizes[3],
                                           l_dim_sizes[1],
                                           l_dim_sizes[0],
                                           l_dim_sizes[5],
                                           l_dim_sizes[8] } );

  at::Tensor l_data_gcexaiy = at::randn( { l_dim_sizes[4],
                                           l_dim_sizes[1],
                                           l_dim_sizes[2],
                                           l_dim_sizes[7],
                                           l_dim_sizes[0],
                                           l_dim_sizes[6],
                                           l_dim_sizes[8] } );

  at::Tensor l_data_xhgfeiy = at::randn( { l_dim_sizes[7],
                                           l_dim_sizes[5],
                                           l_dim_sizes[4],
                                           l_dim_sizes[3],
                                           l_dim_sizes[2],
                                           l_dim_sizes[6],
                                           l_dim_sizes[8] } );


  void * l_data_ptrs[3] = { l_data_fcahy.data_ptr(),
                            l_data_gcexaiy.data_ptr(),
                            l_data_xhgfeiy.data_ptr() };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;
  l_einsum_exp.init( 9,
                     l_dim_sizes,
                     1,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  at::Tensor l_data_xhgfeiy_ref = at::einsum( "fcahy,gcexaiy->xhgfeiy",
                                              { l_data_fcahy,
                                                l_data_gcexaiy } );

  REQUIRE( at::allclose( l_data_xhgfeiy_ref, l_data_xhgfeiy, 1E-3, 1E-6) );
}

TEST_CASE( "Multi-level einsum expression using the internal interface.", "[einsum_exp]" ) {
  // test case:
  //
  // string: iae,bf,dcba,cg,dh->hgfei
  // contraction path: [(1, 2), (2, 3), (0, 1), (0, 1)] 
  //
  //       _______________hgfei______
  //      /                          \
  //   _fcah____                   _gceai_
  //  /         \                 /       \
  // dh        _dcaf_           iae       cg
  //          /      \
  //         bf      dcba
  //
  // char   id   size
  //    a    0      4
  //    b    1      3
  //    c    2      5
  //    d    3      8
  //    e    4      7
  //    f    5      2
  //    g    6      3
  //    h    7      6
  //    i    8      9
  int64_t l_dim_sizes[9] = { 4, 3, 5, 8, 7, 2, 3, 6, 9 };

  int64_t l_string_num_dims[6] = { 3, 2, 4, 2, 2, 5 };

  int64_t l_string_dim_ids[18] = { 8, 0, 4,         // iae
                                   1, 5,            // bf
                                   3, 2, 1, 0,      // dcba
                                   2, 6,            // cg
                                   3, 7,            // dh
                                   7, 6, 5, 4, 8 }; // hgfei

  int64_t l_path[8] = { 1, 2,   // dcaf
                        2, 3,   // fcah
                        0, 1,   // gceai
                        0, 1 }; // hgfei

  // data
  at::Tensor l_data_iae   = at::rand( { l_dim_sizes[8],
                                        l_dim_sizes[0],
                                        l_dim_sizes[4] } );
  at::Tensor l_data_bf    = at::rand( { l_dim_sizes[1] ,
                                        l_dim_sizes[5]} );
  at::Tensor l_data_dcba  = at::rand( { l_dim_sizes[3],
                                        l_dim_sizes[2],
                                        l_dim_sizes[1],
                                        l_dim_sizes[0] } );
  at::Tensor l_data_cg   = at::rand(  { l_dim_sizes[2],
                                        l_dim_sizes[6] } );
  at::Tensor l_data_dh    = at::rand( { l_dim_sizes[3] ,
                                        l_dim_sizes[7]} );
  at::Tensor l_data_hgfei = at::rand( { l_dim_sizes[7],
                                        l_dim_sizes[6],
                                        l_dim_sizes[5],
                                        l_dim_sizes[4],
                                        l_dim_sizes[8] } );

  void * l_data_ptrs[6] = { l_data_iae.data_ptr(),
                            l_data_bf.data_ptr(),
                            l_data_dcba.data_ptr(),
                            l_data_cg.data_ptr(),
                            l_data_dh.data_ptr(),
                            l_data_hgfei.data_ptr() };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;
  l_einsum_exp.init( 9,
                     l_dim_sizes,
                     4,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  at::Tensor l_data_hgfei_ref = at::einsum( "iae,bf,dcba,cg,dh->hgfei",
                                            { l_data_iae,
                                              l_data_bf,
                                              l_data_dcba,
                                              l_data_cg,
                                              l_data_dh } );

  REQUIRE( at::allclose( l_data_hgfei_ref, l_data_hgfei ) );
}

TEST_CASE( "Multi-level einsum expression with C dimensions using the internal interface.", "[einsum_exp]" ) {
  // test case:
  //
  // string: iaxey,ybf,dcba,cxg,dhy->xhgfeiy
  // contraction path: [(1, 2), (2, 3), (0, 1), (0, 1)] 
  //
  //       _____________xhgfeiy______
  //      /                          \
  //   _fcahy___                   _gcexaiy_
  //  /         \                 /         \
  // dhy       _dcafy_           iaxey      cxg
  //          /       \
  //        ybf       dcba
  //
  // char   id   size
  //    a    0      4
  //    b    1      3
  //    c    2      5
  //    d    3      8
  //    e    4      7
  //    f    5      2
  //    g    6      3
  //    h    7      6
  //    i    8      9
  //    x    9      2
  //    y   10      3
  int64_t l_dim_sizes[11] = { 4, 3, 5, 8, 7, 2, 3, 6, 9, 2, 3 };

  int64_t l_string_num_dims[6] = { 5, 3, 4, 3, 3, 7 };

  int64_t l_string_dim_ids[25] = { 8, 0, 9, 4, 10,         // iaxey
                                   10, 1, 5,               // ybf
                                   3, 2, 1, 0,             // dcba
                                   2, 9, 6,                // cxg
                                   3, 7, 10,               // dhy
                                   9, 7, 6, 5, 4, 8, 10 }; // xhgfeiy

  int64_t l_path[8] = { 1, 2,   // dcafy
                        2, 3,   // fcahy
                        0, 1,   // gcexaiy
                        0, 1 }; // xhgfeiy

  // data
  at::Tensor l_data_iaxey   = at::rand( { l_dim_sizes[ 8],
                                          l_dim_sizes[ 0],
                                          l_dim_sizes[ 9],
                                          l_dim_sizes[ 4],
                                          l_dim_sizes[10], } );
  at::Tensor l_data_ybf     = at::rand( { l_dim_sizes[10],
                                          l_dim_sizes[ 1] ,
                                          l_dim_sizes[ 5]} );
  at::Tensor l_data_dcba    = at::rand( { l_dim_sizes[ 3],
                                          l_dim_sizes[ 2],
                                          l_dim_sizes[ 1],
                                          l_dim_sizes[ 0] } );
  at::Tensor l_data_cxg     = at::rand( { l_dim_sizes[ 2],
                                          l_dim_sizes[ 9],
                                          l_dim_sizes[ 6] } );
  at::Tensor l_data_dhy     = at::rand( { l_dim_sizes[ 3] ,
                                          l_dim_sizes[ 7],
                                          l_dim_sizes[10] } );
  at::Tensor l_data_xhgfeiy = at::rand( { l_dim_sizes[ 9],
                                          l_dim_sizes[ 7],
                                          l_dim_sizes[ 6],
                                          l_dim_sizes[ 5],
                                          l_dim_sizes[ 4],
                                          l_dim_sizes[ 8],
                                          l_dim_sizes[10], } );

  void * l_data_ptrs[6] = { l_data_iaxey.data_ptr(),
                            l_data_ybf.data_ptr(),
                            l_data_dcba.data_ptr(),
                            l_data_cxg.data_ptr(),
                            l_data_dhy.data_ptr(),
                            l_data_xhgfeiy.data_ptr() };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;
  l_einsum_exp.init( 11,
                     l_dim_sizes,
                     4,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::FP32,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  at::Tensor l_data_xhgfeiy_ref = at::einsum( "iaxey,ybf,dcba,cxg,dhy->xhgfeiy",
                                              { l_data_iaxey,
                                                l_data_ybf,
                                                l_data_dcba,
                                                l_data_cxg,
                                                l_data_dhy } );

  REQUIRE( at::allclose( l_data_xhgfeiy_ref, l_data_xhgfeiy ) );
}

TEST_CASE( "Multi-level batch-inner complex einsum expression with additional C dimensions using the internal interface.", "[einsum_exp]" ) {
  // test case:
  //
  // string: iaxey,ybf,dcba,cxg,dhy->xhgfeiy
  // contraction path: [(1, 2), (2, 3), (0, 1), (0, 1)] 
  //
  //       _____________xhgfeiy______
  //      /                          \
  //   _fcahy___                   _gcexaiy_
  //  /         \                 /         \
  // dhy       _dcafy_           iaxey      cxg
  //          /       \
  //        ybf       dcba
  //
  // char   id   size
  //    a    0      4
  //    b    1      3
  //    c    2      5
  //    d    3      8
  //    e    4      7
  //    f    5      2
  //    g    6      3
  //    h    7      6
  //    i    8      9
  //    x    9      2
  //    y   10      3
  //    z   11      2 // complex dimension (not given explicitly in tensor names)
  int64_t l_dim_sizes[12] = { 4, 3, 5, 8, 7, 2, 3, 6, 9, 2, 3, 2 };

  int64_t l_string_num_dims[6] = { 6, 4, 5, 4, 4, 8 };

  int64_t l_string_dim_ids[31] = { 8, 0, 9, 4, 10, 11,         // iaxey
                                   10, 1, 5, 11,               // ybf
                                   3, 2, 1, 0, 11,             // dcba
                                   2, 9, 6, 11,                // cxg
                                   3, 7, 10, 11,               // dhy
                                   9, 7, 6, 5, 4, 8, 10, 11 }; // xhgfeiy

  int64_t l_path[8] = { 1, 2,   // dcafy
                        2, 3,   // fcahy
                        0, 1,   // gcexaiy
                        0, 1 }; // xhgfeiy

  // data
  at::Tensor l_data_iaxey   = at::randn( { l_dim_sizes[ 8],
                                           l_dim_sizes[ 0],
                                           l_dim_sizes[ 9],
                                           l_dim_sizes[ 4],
                                           l_dim_sizes[10] },
                                         at::ScalarType::ComplexDouble );
  at::Tensor l_data_ybf     = at::randn( { l_dim_sizes[10],
                                           l_dim_sizes[ 1] ,
                                           l_dim_sizes[ 5]},
                                          at::ScalarType::ComplexDouble );
  at::Tensor l_data_dcba    = at::randn( { l_dim_sizes[ 3],
                                           l_dim_sizes[ 2],
                                           l_dim_sizes[ 1],
                                           l_dim_sizes[ 0] },
                                          at::ScalarType::ComplexDouble );
  at::Tensor l_data_cxg     = at::randn( { l_dim_sizes[ 2],
                                           l_dim_sizes[ 9],
                                           l_dim_sizes[ 6] },
                                          at::ScalarType::ComplexDouble );
  at::Tensor l_data_dhy     = at::randn( { l_dim_sizes[ 3] ,
                                           l_dim_sizes[ 7],
                                           l_dim_sizes[10] },
                                         at::ScalarType::ComplexDouble );
  at::Tensor l_data_xhgfeiy = at::randn( { l_dim_sizes[ 9],
                                           l_dim_sizes[ 7],
                                           l_dim_sizes[ 6],
                                           l_dim_sizes[ 5],
                                           l_dim_sizes[ 4],
                                           l_dim_sizes[ 8],
                                           l_dim_sizes[10] },
                                          at::ScalarType::ComplexDouble );

  void * l_data_ptrs[6] = { l_data_iaxey.data_ptr(),
                            l_data_ybf.data_ptr(),
                            l_data_dcba.data_ptr(),
                            l_data_cxg.data_ptr(),
                            l_data_dhy.data_ptr(),
                            l_data_xhgfeiy.data_ptr() };

  einsum_ir::frontend::EinsumExpression l_einsum_exp;
  l_einsum_exp.init( 12,
                     l_dim_sizes,
                     4,
                     l_string_num_dims,
                     l_string_dim_ids,
                     l_path,
                     einsum_ir::complex_t::BATCH_INNER,
                     einsum_ir::data_t::FP64,
                     l_data_ptrs );

  einsum_ir::err_t l_err = l_einsum_exp.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_einsum_exp.eval();

  // reference
  at::Tensor l_data_xhgfeiy_ref = at::einsum( "iaxey,ybf,dcba,cxg,dhy->xhgfeiy",
                                              { l_data_iaxey,
                                                l_data_ybf,
                                                l_data_dcba,
                                                l_data_cxg,
                                                l_data_dhy } );

  REQUIRE( at::allclose( l_data_xhgfeiy_ref, l_data_xhgfeiy ) );
}