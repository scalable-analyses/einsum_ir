#include <ATen/ATen.h>
#include "catch.hpp"
#include "EinsumNode.h"

TEST_CASE( "Simple matmul example without any intermediate data.", "[einsum_node]" ) {
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
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

  // data
  at::Tensor l_in_left  = at::rand( {4, 2} );
  at::Tensor l_in_right = at::rand( {3, 4} );
  at::Tensor l_out_ref  = at::rand( {3, 2} );
  at::Tensor l_out = l_out_ref.clone();

  // reference
  l_out_ref += at::einsum( "km,nk->nm",
                           {l_in_left, l_in_right} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_0;
  einsum_ir::backend::EinsumNode l_node_1;
  einsum_ir::backend::EinsumNode l_node_2;

  l_node_0.init( 2,
                 l_dim_ids_in_left,
                 &l_dim_sizes,
                 nullptr,
                 einsum_ir::FP32,
                 l_in_left.data_ptr() );

  l_node_1.init( 2,
                 l_dim_ids_in_right,
                 &l_dim_sizes,
                 nullptr,
                 einsum_ir::FP32,
                 l_in_right.data_ptr() );

  l_node_2.init( 2,
                 l_dim_ids_out,
                 &l_dim_sizes,
                 nullptr,
                 nullptr,
                 nullptr,
                 nullptr,
                 einsum_ir::FP32,
                 nullptr,
                 l_out.data_ptr(),
                 einsum_ir::UNDEFINED_KTYPE,
                 einsum_ir::MADD,
                 einsum_ir::UNDEFINED_KTYPE,
                 &l_node_0,
                 &l_node_1 );

  // check node info
  REQUIRE( l_node_0.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_1.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_2.m_data_ptr_ext != nullptr );

  einsum_ir::err_t l_err = l_node_2.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_2.eval();

  // check results
  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "Simple batch-outer complex matmul example without any intermediate data.", "[einsum_node]" ) {
  // test case:
  //
  //    ___cnm___
  //   /         \
  // ckm         cnk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  //    c    3      2
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 2 ) );

  int64_t l_dim_ids_ckm[3] = { 3, 2, 0 };
  int64_t l_dim_ids_cnk[3] = { 3, 1, 2 };
  int64_t l_dim_ids_cnm[3] = { 3, 1, 0 };

  // data
  at::Tensor l_left  = at::randn( {2, 4, 2} );
  at::Tensor l_right = at::randn( {2, 3, 4} );
  at::Tensor l_out   = at::randn( {2, 3, 2} );

  // convert to complex
  at::Tensor l_left_aos    = at::view_as_complex( l_left.permute( {1, 2, 0} ).contiguous() );
  at::Tensor l_right_aos   = at::view_as_complex( l_right.permute( {1, 2, 0} ).contiguous() );
  at::Tensor l_out_ref_aos = at::view_as_complex( l_out.permute( {1, 2, 0} ).contiguous() );

  // reference
  l_out_ref_aos += at::einsum( "km,nk->nm",
                               {l_left_aos, l_right_aos} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_ckm;
  einsum_ir::backend::EinsumNode l_node_cnk;
  einsum_ir::backend::EinsumNode l_node_cnm;

  l_node_ckm.init( 3,
                   l_dim_ids_ckm,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   l_left.data_ptr() );

  l_node_cnk.init( 3,
                   l_dim_ids_cnk,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   l_right.data_ptr() );

  l_node_cnm.init( 3,
                   l_dim_ids_cnm,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   nullptr,
                   l_out.data_ptr(),
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   einsum_ir::kernel_t::CPX_MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &l_node_ckm,
                   &l_node_cnk );

  // check node info
  REQUIRE( l_node_ckm.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_cnk.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_cnm.m_data_ptr_ext != nullptr );

  einsum_ir::err_t l_err = l_node_cnm.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_cnm.eval();

  // check results
  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( {1, 2, 0} ).contiguous() );
  REQUIRE( at::allclose( l_out_aos, l_out_ref_aos )  );
}

TEST_CASE( "Simple complex matmul example with batch-inner input and batch-outer output.", "[einsum_node]" ) {
  // test case:
  //
  //    ___cnm___
  //   /         \
  // kmc         nkc
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  //    c    3      2
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 2 ) );

  int64_t l_dim_ids_kmc[3] = { 2, 0, 3 };
  int64_t l_dim_ids_nkc[3] = { 1, 2, 3 };
  int64_t l_dim_ids_cnm[3] = { 3, 1, 0 };

  // data
  at::Tensor l_left_aos  = at::ones( {4, 2},
                                      at::ScalarType::ComplexFloat );
  at::Tensor l_right_aos = at::ones( {3, 4},
                                      at::ScalarType::ComplexFloat );
  at::Tensor l_out       = at::ones( {2, 3, 2},
                                      at::ScalarType::Float );

  // reference
  at::Tensor l_out_ref = at::einsum( "km,nk->nm",
                                     {l_left_aos, l_right_aos} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_kmc;
  einsum_ir::backend::EinsumNode l_node_nkc;
  einsum_ir::backend::EinsumNode l_node_cnm;

  l_node_kmc.init( 3,
                   l_dim_ids_kmc,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   l_left_aos.data_ptr() );

  l_node_nkc.init( 3,
                   l_dim_ids_nkc,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   l_right_aos.data_ptr() );

  l_node_cnm.init( 3,
                   l_dim_ids_cnm,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   nullptr,
                   l_out.data_ptr(),
                   einsum_ir::kernel_t::CPX_ZERO,
                   einsum_ir::kernel_t::CPX_MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &l_node_kmc,
                   &l_node_nkc );

  // check node info
  REQUIRE( l_node_kmc.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_nkc.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_cnm.m_data_ptr_ext != nullptr );

  einsum_ir::err_t l_err = l_node_cnm.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_cnm.eval();

  // check results
  at::Tensor l_out_aos = at::view_as_complex( l_out.permute( {1, 2, 0} ).contiguous() );

  REQUIRE( at::allclose( l_out_aos, l_out_ref )  );
}

TEST_CASE( "Simple complex matmul example with batch-inner input and batch-inner output.", "[einsum_node]" ) {
  // test case:
  //
  //       nmc
  //        |
  //    ___cnm___
  //   /         \
  // kmc         nkc
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  //    c    3      2
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 2 ) );

  int64_t l_dim_ids_kmc[3] = { 2, 0, 3 };
  int64_t l_dim_ids_nkc[3] = { 1, 2, 3 };
  int64_t l_dim_ids_cnm[3] = { 3, 1, 0 };
  int64_t l_dim_ids_nmc[3] = { 1, 0, 3 };

  // data
  at::Tensor l_data_kmc = at::randn( {4, 2},
                                      at::ScalarType::ComplexFloat );
  at::Tensor l_data_nkc = at::randn( {3, 4},
                                      at::ScalarType::ComplexFloat );
  at::Tensor l_data_nmc = at::randn( {3, 2},
                                      at::ScalarType::ComplexFloat );

  // reference
  at::Tensor l_data_nmc_ref = at::einsum( "km,nk->nm",
                                          {l_data_kmc, l_data_nkc} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_kmc;
  einsum_ir::backend::EinsumNode l_node_nkc;
  einsum_ir::backend::EinsumNode l_node_cnm;
  einsum_ir::backend::EinsumNode l_node_nmc;

  l_node_kmc.init( 3,
                   l_dim_ids_kmc,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   l_data_kmc.data_ptr() );

  l_node_nkc.init( 3,
                   l_dim_ids_nkc,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   l_data_nkc.data_ptr() );

  l_node_cnm.init( 3,
                   l_dim_ids_cnm,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   nullptr,
                   nullptr,
                   einsum_ir::kernel_t::CPX_ZERO,
                   einsum_ir::kernel_t::CPX_MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &l_node_kmc,
                   &l_node_nkc );

  l_node_nmc.init( 3,
                   l_dim_ids_nmc,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::data_t::FP32,
                   l_data_nmc.data_ptr(),
                   &l_node_cnm );

  einsum_ir::err_t l_err = l_node_nmc.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_nmc.eval();

  // check results
  REQUIRE( at::allclose( l_data_nmc, l_data_nmc_ref )  );
}

TEST_CASE( "Two matmul example with external intermediate data.", "[einsum_node]" ) {
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
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 5 ) );

  int64_t l_dim_ids_ca[2] = { 2, 0 };
  int64_t l_dim_ids_bc[2] = { 1, 2 };
  int64_t l_dim_ids_da[2] = { 3, 0 };
  int64_t l_dim_ids_ba[2] = { 1, 0 };
  int64_t l_dim_ids_bd[2] = { 1, 3 };

  // data
  at::Tensor l_data_ca     = at::rand(  {4, 2} );
  at::Tensor l_data_bc     = at::rand(  {3, 4} );
  at::Tensor l_data_da     = at::rand(  {5, 2} );
  at::Tensor l_data_ba     = at::zeros( {3, 2} );
  at::Tensor l_data_bd_ref = at::rand(  {3, 5} );
  at::Tensor l_data_bd     = l_data_bd_ref.clone();

  // reference
  l_data_bd_ref = at::einsum( "ca,bc,da->bd",
                              {l_data_ca, l_data_bc, l_data_da} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_ca;
  einsum_ir::backend::EinsumNode l_node_bc;
  einsum_ir::backend::EinsumNode l_node_da;
  einsum_ir::backend::EinsumNode l_node_ba;
  einsum_ir::backend::EinsumNode l_node_bd;

  l_node_ca.init( 2,
                  l_dim_ids_ca,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_ca.data_ptr() );

  l_node_bc.init( 2,
                  l_dim_ids_bc,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_bc.data_ptr() );

  l_node_da.init( 2,
                  l_dim_ids_da,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_da.data_ptr() );

  l_node_ba.init( 2,
                  l_dim_ids_ba,
                  &l_dim_sizes,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  einsum_ir::FP32,
                  nullptr,
                  l_data_ba.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  &l_node_ca,
                  &l_node_bc );

  l_node_bd.init( 2,
                  l_dim_ids_bd,
                  &l_dim_sizes,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  einsum_ir::FP32,
                  nullptr,
                  l_data_bd.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  &l_node_ba,
                  &l_node_da );

  einsum_ir::err_t l_err = l_node_bd.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // check node info
  REQUIRE( l_node_ca.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_bc.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_da.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_ba.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_bd.m_data_ptr_ext != nullptr );

  l_node_bd.eval();

  // check results
  REQUIRE( at::allclose( l_data_bd, l_data_bd_ref )  );
}

TEST_CASE( "Two matmul example with locked data.", "[einsum_node]" ) {
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
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 5 ) );

  int64_t l_dim_ids_ca[2] = { 2, 0 };
  int64_t l_dim_ids_bc[2] = { 1, 2 };
  int64_t l_dim_ids_da[2] = { 3, 0 };
  int64_t l_dim_ids_ba[2] = { 1, 0 };
  int64_t l_dim_ids_bd[2] = { 1, 3 };

  // data
  at::Tensor l_data_ca     = at::rand(  {4, 2} );
  at::Tensor l_data_bc     = at::rand(  {3, 4} );
  at::Tensor l_data_da     = at::rand(  {5, 2} );
  at::Tensor l_data_bd_ref = at::rand(  {3, 5} );
  at::Tensor l_data_bd     = l_data_bd_ref.clone();

  // reference
  l_data_bd_ref = at::einsum( "ca,bc,da->bd",
                              {l_data_ca, l_data_bc, l_data_da} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_ca;
  einsum_ir::backend::EinsumNode l_node_bc;
  einsum_ir::backend::EinsumNode l_node_da;
  einsum_ir::backend::EinsumNode l_node_ba;
  einsum_ir::backend::EinsumNode l_node_bd;

  l_node_ca.init( 2,
                  l_dim_ids_ca,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_ca.data_ptr() );

  l_node_bc.init( 2,
                  l_dim_ids_bc,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_bc.data_ptr() );

  l_node_da.init( 2,
                  l_dim_ids_da,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_da.data_ptr() );

  l_node_ba.init( 2,
                  l_dim_ids_ba,
                  &l_dim_sizes,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  einsum_ir::FP32,
                  nullptr,
                  nullptr,
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  &l_node_ca,
                  &l_node_bc );

  l_node_bd.init( 2,
                  l_dim_ids_bd,
                  &l_dim_sizes,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  einsum_ir::FP32,
                  nullptr,
                  l_data_bd.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  &l_node_ba,
                  &l_node_da );

  einsum_ir::err_t l_err = l_node_bd.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_err = l_node_ca.store_and_lock_data();
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_node_bc.store_and_lock_data();
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_node_da.store_and_lock_data();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // check node info
  REQUIRE( l_node_ca.m_data_ptr_int != nullptr );
  REQUIRE( l_node_bc.m_data_ptr_int != nullptr );
  REQUIRE( l_node_da.m_data_ptr_int != nullptr );

  // modify external data
  l_data_ca += at::rand( {4, 2} );
  l_data_bc += at::rand( {3, 4} );
  l_data_da += at::rand( {5, 2} );

  l_node_bd.eval();

  // check results
  REQUIRE( at::allclose( l_data_bd, l_data_bd_ref )  );

  // unlock data
  l_err = l_node_ca.unlock_data();
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_node_bc.unlock_data();
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_node_da.unlock_data();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_bd.eval();

  // reference
  l_data_bd_ref = at::einsum( "ca,bc,da->bd",
                              {l_data_ca, l_data_bc, l_data_da} );

  // check results
  REQUIRE( at::allclose( l_data_bd, l_data_bd_ref )  );
}

TEST_CASE( "Two matmul example with internal intermediate data.", "[einsum_node]" ) {
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
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 5 ) );

  int64_t l_dim_ids_ca[2] = { 2, 0 };
  int64_t l_dim_ids_bc[2] = { 1, 2 };
  int64_t l_dim_ids_da[2] = { 3, 0 };
  int64_t l_dim_ids_ba[2] = { 1, 0 };
  int64_t l_dim_ids_bd[2] = { 1, 3 };

  // data
  at::Tensor l_data_ca     = at::rand(  {4, 2} );
  at::Tensor l_data_bc     = at::rand(  {3, 4} );
  at::Tensor l_data_da     = at::rand(  {5, 2} );
  at::Tensor l_data_bd_ref = at::rand(  {3, 5} );
  at::Tensor l_data_bd     = l_data_bd_ref.clone();

  // reference
  l_data_bd_ref = at::einsum( "ca,bc,da->bd",
                              {l_data_ca, l_data_bc, l_data_da} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_ca;
  einsum_ir::backend::EinsumNode l_node_bc;
  einsum_ir::backend::EinsumNode l_node_da;
  einsum_ir::backend::EinsumNode l_node_ba;
  einsum_ir::backend::EinsumNode l_node_bd;

  l_node_ca.init( 2,
                  l_dim_ids_ca,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_ca.data_ptr() );

  l_node_bc.init( 2,
                  l_dim_ids_bc,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_bc.data_ptr() );

  l_node_da.init( 2,
                  l_dim_ids_da,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_da.data_ptr() );

  l_node_ba.init( 2,
                  l_dim_ids_ba,
                  &l_dim_sizes,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  einsum_ir::FP32,
                  nullptr,
                  nullptr,
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  &l_node_ca,
                  &l_node_bc );

  l_node_bd.init( 2,
                  l_dim_ids_bd,
                  &l_dim_sizes,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  einsum_ir::FP32,
                  nullptr,
                  l_data_bd.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  &l_node_ba,
                  &l_node_da );

  einsum_ir::err_t l_err = l_node_bd.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_bd.eval();

  // check results
  REQUIRE( at::allclose( l_data_bd, l_data_bd_ref )  );
}

TEST_CASE( "Complex two matmul example with batch-inner input data and batch-outer internal intermediate data.", "[einsum_node]" ) {
  // test case:
  //
  //           bdx
  //            |
  //         __xbd__
  //        /       \
  //    __bax___    dax
  //   /        \
  // cax          bcx
  //
  // char   id   size
  //    a    0      2
  //    b    1      3
  //    c    2      4
  //    d    3      5
  //    x    4      2 // complex dimension
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4, 2 ) );

  int64_t l_dim_ids_cax[3] = { 2, 0, 4 };
  int64_t l_dim_ids_bcx[3] = { 1, 2, 4 };
  int64_t l_dim_ids_dax[3] = { 3, 0, 4 };
  int64_t l_dim_ids_bax[3] = { 1, 0, 4 };
  int64_t l_dim_ids_xbd[3] = { 4, 1, 3 };
  int64_t l_dim_ids_bdx[3] = { 1, 3, 4 };

  // data
  at::Tensor l_data_cax     = at::randn( {4, 2},
                                         at::ScalarType::ComplexFloat );
  at::Tensor l_data_bcx     = at::randn( {3, 4},
                                         at::ScalarType::ComplexFloat );
  at::Tensor l_data_dax     = at::randn( {5, 2},
                                         at::ScalarType::ComplexFloat );
  at::Tensor l_data_bdx     = at::randn( {3, 5},
                                         at::ScalarType::ComplexFloat );

  // reference
  at::Tensor l_data_bdx_ref = at::einsum( "ca,bc,da->bd",
                                         {l_data_cax, l_data_bcx, l_data_dax} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_cax;
  einsum_ir::backend::EinsumNode l_node_bcx;
  einsum_ir::backend::EinsumNode l_node_dax;
  einsum_ir::backend::EinsumNode l_node_bax;
  einsum_ir::backend::EinsumNode l_node_xbd;
  einsum_ir::backend::EinsumNode l_node_bdx;

  l_node_cax.init( 3,
                   l_dim_ids_cax,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::FP32,
                   l_data_cax.data_ptr() );

  l_node_bcx.init( 3,
                   l_dim_ids_bcx,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::FP32,
                   l_data_bcx.data_ptr() );

  l_node_dax.init( 3,
                   l_dim_ids_dax,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::FP32,
                   l_data_dax.data_ptr() );

  l_node_bax.init( 3,
                   l_dim_ids_bax,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   einsum_ir::FP32,
                   nullptr,
                   nullptr,
                   einsum_ir::CPX_ZERO,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE,
                   &l_node_cax,
                   &l_node_bcx );

  l_node_xbd.init( 3,
                   l_dim_ids_xbd,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   einsum_ir::FP32,
                   nullptr,
                   nullptr,
                   einsum_ir::CPX_ZERO,
                   einsum_ir::CPX_MADD,
                   einsum_ir::UNDEFINED_KTYPE,
                   &l_node_bax,
                   &l_node_dax );

  l_node_bdx.init( 3,
                   l_dim_ids_bdx,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::FP32,
                   l_data_bdx.data_ptr(),
                   &l_node_xbd );

  einsum_ir::err_t l_err = l_node_bdx.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_bdx.eval();

  // check results
  REQUIRE( at::allclose( l_data_bdx, l_data_bdx_ref )  );
}

TEST_CASE( "Matmul example possibly requiring permuted input data.", "[einsum_node]" ) {
  // test case:
  //
  //    ____nm___
  //   /         \
  // mk           kn
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_mk[2] = { 0, 2 };
  int64_t l_dim_ids_kn[2] = { 2, 1 };
  int64_t l_dim_ids_nm[2] = { 1, 0 };

  // data
  at::Tensor l_data_mk     = at::rand( {2, 4} );
  at::Tensor l_data_kn     = at::rand( {4, 3} );
  at::Tensor l_data_nm_ref = at::rand( {3, 2} );
  at::Tensor l_data_nm     = l_data_nm_ref.clone();

  // reference
  l_data_nm_ref = at::einsum( "mk,kn->nm",
                              {l_data_mk, l_data_kn} );

  // einsum_ir
  einsum_ir::backend::EinsumNode l_node_mk;
  einsum_ir::backend::EinsumNode l_node_kn;
  einsum_ir::backend::EinsumNode l_node_nm;

  l_node_mk.init( 2,
                  l_dim_ids_mk,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_mk.data_ptr() );

  l_node_kn.init( 2,
                  l_dim_ids_kn,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_kn.data_ptr() );

  l_node_nm.init( 2,
                  l_dim_ids_nm,
                  &l_dim_sizes,
                  nullptr,
                  nullptr,
                  nullptr,
                  nullptr,
                  einsum_ir::FP32,
                  nullptr,
                  l_data_nm.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  &l_node_mk,
                  &l_node_kn );

  // compile
  einsum_ir::err_t l_err = l_node_nm.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // check node info
  REQUIRE( l_node_mk.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_kn.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_nm.m_data_ptr_ext != nullptr );

  REQUIRE( l_node_mk.m_data_ptr_int != nullptr );
  REQUIRE( l_node_kn.m_data_ptr_int != nullptr );
  REQUIRE( l_node_nm.m_data_ptr_int == nullptr );

  l_node_nm.eval();

  // check results
  REQUIRE( at::allclose( l_data_nm_ref, l_data_nm )  );
}

TEST_CASE( "Einsum expression without batch dimensions.", "[einsum_node]" ) {
  // test case:
  //
  //        _______________iefgh______
  //      /                          \
  //    _hacf____                   _iaecg_
  //  /         \                 /        \
  //  hd        _facd_           eai       gic
  //          /       \
  //         fb      abcd
  //
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'b', 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'c', 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'd', 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'e', 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'f', 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'g', 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'h', 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 'i', 2 ) );

  int64_t l_dim_ids_hd[2]    = { 'h', 'd' };
  int64_t l_dim_ids_fb[2]    = { 'f', 'b' };
  int64_t l_dim_ids_abcd[4]  = { 'a', 'b', 'c', 'd' };
  int64_t l_dim_ids_eai[3]   = { 'e', 'a', 'i' };
  int64_t l_dim_ids_gic[3]   = { 'g', 'i', 'c' };
  int64_t l_dim_ids_hacf[4]  = { 'h', 'a', 'c', 'f' };
  int64_t l_dim_ids_facd[4]  = { 'f', 'a', 'c', 'd' };
  int64_t l_dim_ids_iaecg[5] = { 'i', 'a', 'e', 'c', 'g' };
  int64_t l_dim_ids_iefgh[5] = { 'i', 'e', 'f', 'g', 'h' };

  std::vector< int64_t > l_sizes_hd;
  for( int64_t l_di = 0; l_di < 2; l_di++ ) {
    l_sizes_hd.push_back( l_dim_sizes.at( l_dim_ids_hd[l_di] ) );
  }

  std::vector< int64_t > l_sizes_fb;
  for( int64_t l_di = 0; l_di < 2; l_di++ ) {
    l_sizes_fb.push_back( l_dim_sizes.at( l_dim_ids_fb[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abcd;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_abcd.push_back( l_dim_sizes.at( l_dim_ids_abcd[l_di] ) );
  }

  std::vector< int64_t > l_sizes_eai;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_eai.push_back( l_dim_sizes.at( l_dim_ids_eai[l_di] ) );
  }

  std::vector< int64_t > l_sizes_gic;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_gic.push_back( l_dim_sizes.at( l_dim_ids_gic[l_di] ) );
  }

  std::vector< int64_t > l_sizes_hacf;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_hacf.push_back( l_dim_sizes.at( l_dim_ids_hacf[l_di] ) );
  }

  std::vector< int64_t > l_sizes_iaecg;
  for( int64_t l_di = 0; l_di < 5; l_di++ ) {
    l_sizes_iaecg.push_back( l_dim_sizes.at( l_dim_ids_iaecg[l_di] ) );
  }

  std::vector< int64_t > l_sizes_iefgh;
  for( int64_t l_di = 0; l_di < 5; l_di++ ) {
    l_sizes_iefgh.push_back( l_dim_sizes.at( l_dim_ids_iefgh[l_di] ) );
  }

  // data
  at::Tensor l_data_hd        = at::rand( l_sizes_hd );
  at::Tensor l_data_fb        = at::rand( l_sizes_fb );
  at::Tensor l_data_abcd      = at::rand( l_sizes_abcd );
  at::Tensor l_data_eai       = at::rand( l_sizes_eai );
  at::Tensor l_data_gic       = at::rand( l_sizes_gic );
  at::Tensor l_data_iefgh_ref = at::rand( l_sizes_iefgh );
  at::Tensor l_data_iefgh     = l_data_iefgh_ref.clone();

  einsum_ir::backend::EinsumNode l_node_hd;
  einsum_ir::backend::EinsumNode l_node_fb;
  einsum_ir::backend::EinsumNode l_node_abcd;
  einsum_ir::backend::EinsumNode l_node_eai;
  einsum_ir::backend::EinsumNode l_node_gic;
  einsum_ir::backend::EinsumNode l_node_hacf;
  einsum_ir::backend::EinsumNode l_node_facd;
  einsum_ir::backend::EinsumNode l_node_iaecg;
  einsum_ir::backend::EinsumNode l_node_iefgh;

  // leaf nodes
  l_node_hd.init( 2,
                  l_dim_ids_hd,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_hd.data_ptr() );

  l_node_fb.init( 2,
                  l_dim_ids_fb,
                  &l_dim_sizes,
                  nullptr,
                  einsum_ir::FP32,
                  l_data_fb.data_ptr() );

  l_node_abcd.init( 4,
                    l_dim_ids_abcd,
                    &l_dim_sizes,
                    nullptr,
                    einsum_ir::FP32,
                    l_data_abcd.data_ptr() );

  l_node_eai.init( 3,
                   l_dim_ids_eai,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::FP32,
                   l_data_eai.data_ptr() );

  l_node_gic.init( 3,
                   l_dim_ids_gic,
                   &l_dim_sizes,
                   nullptr,
                   einsum_ir::FP32,
                   l_data_gic.data_ptr() );

  // dependent nodes
  l_node_hacf.init( 4,
                    l_dim_ids_hacf,
                    &l_dim_sizes,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    einsum_ir::FP32,
                    nullptr,
                    nullptr,
                    einsum_ir::ZERO,
                    einsum_ir::MADD,
                    einsum_ir::UNDEFINED_KTYPE,
                    &l_node_hd,
                    &l_node_facd );

  l_node_facd.init( 4,
                    l_dim_ids_facd,
                    &l_dim_sizes,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    einsum_ir::FP32,
                    nullptr,
                    nullptr,
                    einsum_ir::ZERO,
                    einsum_ir::MADD,
                    einsum_ir::UNDEFINED_KTYPE,
                    &l_node_fb,
                    &l_node_abcd );

  l_node_iaecg.init( 5,
                     l_dim_ids_iaecg,
                     &l_dim_sizes,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     einsum_ir::FP32,
                     nullptr,
                     nullptr,
                     einsum_ir::ZERO,
                     einsum_ir::MADD,
                     einsum_ir::UNDEFINED_KTYPE,
                     &l_node_eai,
                     &l_node_gic );

  l_node_iefgh.init( 5,
                     l_dim_ids_iefgh,
                     &l_dim_sizes,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     einsum_ir::FP32,
                     nullptr,
                     l_data_iefgh.data_ptr(),
                     einsum_ir::ZERO,
                     einsum_ir::MADD,
                     einsum_ir::UNDEFINED_KTYPE,
                     &l_node_hacf,
                     &l_node_iaecg );

  einsum_ir::err_t l_err = l_node_iefgh.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_iefgh.eval();

  // reference
  l_data_iefgh_ref = at::einsum( "hd,fb,abcd,eai,gic->iefgh",
                                 {l_data_hd, l_data_fb, l_data_abcd, l_data_eai, l_data_gic} );


  REQUIRE( at::allclose( l_data_iefgh_ref, l_data_iefgh ) );
}