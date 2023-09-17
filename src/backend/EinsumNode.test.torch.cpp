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
  REQUIRE( l_node_0.m_data_ptr_ext != nullptr );
  REQUIRE( l_node_2.m_data_ptr_ext != nullptr );

  l_node_2.compile();

  l_node_2.eval();

  // check results
  REQUIRE( at::allclose( l_out, l_out_ref )  );
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

  l_node_bd.compile();

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

  l_node_bd.compile();

  l_node_bd.eval();

  // check results
  REQUIRE( at::allclose( l_data_bd, l_data_bd_ref )  );
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
  l_node_nm.compile();

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
  //  /         \                 /       \
  //  hd        _facd_           eai       gic
  //          /      \
  //          fb      abcd
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

  l_node_iefgh.compile();
  l_node_iefgh.eval();

  // reference
  l_data_iefgh_ref = at::einsum( "hd,fb,abcd,eai,gic->iefgh",
                                 {l_data_hd, l_data_fb, l_data_abcd, l_data_eai, l_data_gic} );


  REQUIRE( at::allclose( l_data_iefgh_ref, l_data_iefgh ) );
}

TEST_CASE( "Einsum expression performing a single convolution with a broadcasted bias.", "[einsum_node]" ) {
  // test case:
  //
  //    ___fab____
  //   /          \
  // eab         fecd
  //
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'a', 12 ) ); // height
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'b', 13 ) ); // width
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'c',  3 ) ); // kernel (height)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'd',  3 ) ); // kernel (width)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'e',  5 ) ); // input features
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'f',  9 ) ); // output features

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'a', 12+2 ) ); // height
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'b', 13+2 ) ); // width
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'c',    3 ) ); // kernel (height)
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'd',    3 ) ); // kernel (width)
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'e',    5 ) ); // input features
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'f',    9 ) ); // output features

  std::map< int64_t, int64_t > l_dim_sizes_aux_outer;
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'a', 1 ) ); // height
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'b', 1 ) ); // width
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'f', 9 ) ); // output features

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'c', 'a' ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'd', 'b' ) );

  int64_t l_dim_ids_eab[3]  = { 'e', 'a', 'b' };
  int64_t l_dim_ids_fecd[4] = { 'f', 'e', 'c', 'd' };
  int64_t l_dim_ids_fab[3]  = { 'f', 'a', 'b' };

  std::vector< int64_t > l_sizes_eab;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_eab.push_back( l_dim_sizes_outer.at( l_dim_ids_eab[l_di] ) );
  }

  std::vector< int64_t > l_sizes_fecd;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_fecd.push_back( l_dim_sizes_outer.at( l_dim_ids_fecd[l_di] ) );
  }

  std::vector< int64_t > l_sizes_fab;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_fab.push_back( l_dim_sizes_outer.at( l_dim_ids_fab[l_di] ) );
  }

  std::vector< int64_t > l_sizes_fab_aux;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_fab_aux.push_back( l_dim_sizes_aux_outer.at( l_dim_ids_fab[l_di] ) );
  }

  at::Tensor l_data_eab     = at::randn( l_sizes_eab     );
  at::Tensor l_data_fecd    = at::randn( l_sizes_fecd    );
  at::Tensor l_data_fab_aux = at::randn( l_sizes_fab_aux );
  at::Tensor l_data_fab     = at::randn( l_sizes_fab     );

  einsum_ir::backend::EinsumNode l_node_eab;
  einsum_ir::backend::EinsumNode l_node_fecd;
  einsum_ir::backend::EinsumNode l_node_fab;

  // leaf nodes
  l_node_eab.init( 3,
                   l_dim_ids_eab,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_outer,
                   einsum_ir::FP32,
                   l_data_eab.data_ptr() );

  l_node_fecd.init( 4,
                    l_dim_ids_fecd,
                    &l_dim_sizes_inner,
                    &l_dim_sizes_outer,
                    einsum_ir::FP32,
                    l_data_fecd.data_ptr() );

  // dependent node
  l_node_fab.init( 3,
                   l_dim_ids_fab,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_aux_outer,
                   &l_dim_sizes_outer,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   &l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   l_data_fab_aux.data_ptr(),
                   l_data_fab[0][1][1].data_ptr(),
                   einsum_ir::kernel_t::COPY,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::RELU,
                   &l_node_eab,
                   &l_node_fecd );

  // compile
  einsum_ir::err_t l_err = l_node_fab.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_fab.eval();

  // reference
  at::Tensor l_data_fab_ref = at::conv2d( l_data_eab,
                                          l_data_fecd );
  l_data_fab_ref += l_data_fab_aux;
  l_data_fab_ref = at::relu( l_data_fab_ref );

  // remove padding for comparison
  at::Tensor l_data_fab_narrow = l_data_fab.narrow( 1, 1, 12 ).narrow( 2, 1, 13 );

  REQUIRE( at::allclose( l_data_fab_ref, l_data_fab_narrow, 1E-3, 1E-6 ) );
}

TEST_CASE( "Einsum expression performing a single convolution with a broadcasted bias. Output features are M.", "[einsum_node]" ) {
  // test case:
  //
  //    ___fab____
  //   /          \
  // abe         fecd
  //
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'a', 12 ) ); // height
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'b', 13 ) ); // width
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'c',  3 ) ); // kernel (height)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'd',  3 ) ); // kernel (width)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'e',  5 ) ); // input features
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'f',  9 ) ); // output features

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'a', 12+2 ) ); // height
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'b', 13+2 ) ); // width
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'c',    3 ) ); // kernel (height)
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'd',    3 ) ); // kernel (width)
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'e',    5 ) ); // input features
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'f',    9 ) ); // output features

  std::map< int64_t, int64_t > l_dim_sizes_aux_outer;
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'a', 1 ) ); // height
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'b', 1 ) ); // width
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'f', 9 ) ); // output features

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'c', 'a' ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'd', 'b' ) );

  int64_t l_dim_ids_abe[3]  = { 'a', 'b', 'e' };
  int64_t l_dim_ids_fecd[4] = { 'f', 'e', 'c', 'd' };
  int64_t l_dim_ids_abf[3]  = { 'a', 'b', 'f' };

  std::vector< int64_t > l_sizes_abe;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abe.push_back( l_dim_sizes_outer.at( l_dim_ids_abe[l_di] ) );
  }

  std::vector< int64_t > l_sizes_fecd;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_fecd.push_back( l_dim_sizes_outer.at( l_dim_ids_fecd[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abf;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abf.push_back( l_dim_sizes_outer.at( l_dim_ids_abf[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abf_aux;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abf_aux.push_back( l_dim_sizes_aux_outer.at( l_dim_ids_abf[l_di] ) );
  }

  at::Tensor l_data_abe     = at::randn( l_sizes_abe     );
  at::Tensor l_data_fecd    = at::randn( l_sizes_fecd    );
  at::Tensor l_data_abf_aux = at::randn( l_sizes_abf_aux );
  at::Tensor l_data_abf     = at::randn( l_sizes_abf     );

  einsum_ir::backend::EinsumNode l_node_abe;
  einsum_ir::backend::EinsumNode l_node_fecd;
  einsum_ir::backend::EinsumNode l_node_abf;

  // leaf nodes
  l_node_abe.init( 3,
                   l_dim_ids_abe,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_outer,
                   einsum_ir::FP32,
                   l_data_abe.data_ptr() );

  l_node_fecd.init( 4,
                    l_dim_ids_fecd,
                    &l_dim_sizes_inner,
                    &l_dim_sizes_outer,
                    einsum_ir::FP32,
                    l_data_fecd.data_ptr() );

  // dependent node
  l_node_abf.init( 3,
                   l_dim_ids_abf,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_aux_outer,
                   &l_dim_sizes_outer,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   &l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   l_data_abf_aux.data_ptr(),
                   l_data_abf[1][1][0].data_ptr(),
                   einsum_ir::kernel_t::COPY,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::RELU,
                   &l_node_abe,
                   &l_node_fecd );

  // compile
  einsum_ir::err_t l_err = l_node_abf.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  l_node_abf.eval();

  // reference
  at::Tensor l_data_fab_ref = at::conv2d( l_data_abe.permute( {2, 0, 1} ),
                                          l_data_fecd );
  l_data_fab_ref += l_data_abf_aux.permute( {2, 0, 1} );
  l_data_fab_ref = at::relu( l_data_fab_ref );

  // remove padding for comparison
  at::Tensor l_data_fab_narrow = l_data_abf.permute( {2, 0, 1} ).narrow( 1, 1, 12 ).narrow( 2, 1, 13 );

  REQUIRE( at::allclose( l_data_fab_ref, l_data_fab_narrow, 1E-3, 1E-6 ) );
}

TEST_CASE( "Einsum expression performing a dimension-preserving ResNet-18 block.", "[einsum_node]" ) {
  // layer1 of torchvision.models.resnet18( weights = "ResNet18_Weights.DEFAULT" )
  //
  // Sequential(
  //   (0): BasicBlock(
  //     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  //     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //     (relu): ReLU(inplace=True)
  //     (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  //     (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //   )
  //   (1): BasicBlock(
  //     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  //     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //     (relu): ReLU(inplace=True)
  //     (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  //     (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //   )
  // )

  // corresponding einsum-like tree:
  //
  //           abi
  //            | 
  //         ___+____
  //        /         \
  //       /      ____abi____
  //      /      /           \
  //     /  ___abf____      ifgh
  //     | /          \
  //     abe         fecd
  //
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'a',   28 ) ); // height
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'b',   28 ) ); // width
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'c',    3 ) ); // first kernel (height)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'd',    3 ) ); // first kernel (width)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'e',   64 ) ); // input features
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'f',   64 ) ); // intermediate features
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'g',    3 ) ); // second kernel (height)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'h',    3 ) ); // second kernel (width)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'i',   64 ) ); // output features

  std::map< int64_t, int64_t > l_dim_sizes_outer;
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'a', 28+2 ) ); // height
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'b', 28+2 ) ); // width
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'c',    3 ) ); // first kernel (height)
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'd',    3 ) ); // first kernel (width)
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'e',   64 ) ); // input features
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'f',   64 ) ); // intermediate features
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'g',    3 ) ); // second kernel (height)
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'h',    3 ) ); // second kernel (width)
  l_dim_sizes_outer.insert( std::pair< int64_t, int64_t >( 'i',   64 ) ); // output features

  std::map< int64_t, int64_t > l_dim_sizes_aux_outer;
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'a',  1 ) ); // height
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'b',  1 ) ); // width
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'f', 64 ) ); // intermediate features
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'i', 64 ) ); // otuput features

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'c', 'a' ) ); // first convolution, height
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'd', 'b' ) ); // first convolution, width
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'g', 'a' ) ); // second convolution, height
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'h', 'b' ) ); // second convolution, width

  std::map< int64_t, int64_t > l_offsets;
  l_offsets.insert( std::pair< int64_t, int64_t >( 'a',  1 ) );
  l_offsets.insert( std::pair< int64_t, int64_t >( 'b',  1 ) );

  int64_t l_dim_ids_abe[3]  = { 'a', 'b', 'e' };
  int64_t l_dim_ids_fecd[4] = { 'f', 'e', 'c', 'd' };
  int64_t l_dim_ids_abf[3]  = { 'a', 'b', 'f' };
  int64_t l_dim_ids_ifgh[4] = { 'i', 'f', 'g', 'h' };
  int64_t l_dim_ids_abi[3]  = { 'a', 'b', 'i' };

  std::vector< int64_t > l_sizes_abe;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abe.push_back( l_dim_sizes_outer.at( l_dim_ids_abe[l_di] ) );
  }

  std::vector< int64_t > l_sizes_fecd;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_fecd.push_back( l_dim_sizes_outer.at( l_dim_ids_fecd[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abf;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abf.push_back( l_dim_sizes_outer.at( l_dim_ids_abf[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abf_aux;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abf_aux.push_back( l_dim_sizes_aux_outer.at( l_dim_ids_abf[l_di] ) );
  }

  std::vector< int64_t > l_sizes_ifgh;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_ifgh.push_back( l_dim_sizes_outer.at( l_dim_ids_ifgh[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abi_aux;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abi_aux.push_back( l_dim_sizes_aux_outer.at( l_dim_ids_abi[l_di] ) );
  }

  at::Tensor l_data_abe     = at::zeros( l_sizes_abe    );
  at::Tensor l_data_fecd    = at::randn( l_sizes_fecd    );
  at::Tensor l_data_abf_aux = at::randn( l_sizes_abf_aux );
  at::Tensor l_data_abf     = at::zeros( l_sizes_abf    );
  at::Tensor l_data_ifgh    = at::randn( l_sizes_ifgh    );
  at::Tensor l_data_abi_aux = at::randn( l_sizes_abi_aux );

  einsum_ir::backend::EinsumNode l_node_abe;
  einsum_ir::backend::EinsumNode l_node_fecd;
  einsum_ir::backend::EinsumNode l_node_abf;
  einsum_ir::backend::EinsumNode l_node_ifgh;
  einsum_ir::backend::EinsumNode l_node_abi;

  // leaf nodes
  l_node_abe.init( 3,
                   l_dim_ids_abe,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_outer,
                   einsum_ir::FP32,
                   l_data_abe.data_ptr() );

  l_node_fecd.init( 4,
                    l_dim_ids_fecd,
                    &l_dim_sizes_inner,
                    &l_dim_sizes_outer,
                    einsum_ir::FP32,
                    l_data_fecd.data_ptr() );

  l_node_ifgh.init( 4,
                    l_dim_ids_ifgh,
                    &l_dim_sizes_inner,
                    &l_dim_sizes_outer,
                    einsum_ir::FP32,
                    l_data_ifgh.data_ptr() );

  // dependent nodes
  l_node_abf.init( 3,
                   l_dim_ids_abf,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_aux_outer,
                   &l_dim_sizes_outer,
                   nullptr,
                   &l_offsets,
                   nullptr,
                   nullptr,
                   nullptr,
                   &l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   l_data_abf_aux.data_ptr(),
                   l_data_abf.data_ptr(),
                   einsum_ir::kernel_t::COPY,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::RELU,
                   &l_node_abe,
                   &l_node_fecd );

  l_node_abi.init( 3,
                   l_dim_ids_abi,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_aux_outer,
                   &l_dim_sizes_outer,
                   nullptr,
                   &l_offsets,
                   nullptr,
                   nullptr,
                   nullptr,
                   &l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   l_data_abi_aux.data_ptr(),
                   l_data_abe.data_ptr(),
                   einsum_ir::kernel_t::ADD,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &l_node_abf,
                   &l_node_ifgh );

  // compile
  einsum_ir::err_t l_err = l_node_abi.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // copy in the filters
  l_node_fecd.store_and_lock_data();
  l_node_ifgh.store_and_lock_data();

  // assign input data
  at::Tensor l_data_abe_no_pad = l_data_abe.narrow( 0, 1, 28 );
  l_data_abe_no_pad = l_data_abe_no_pad.narrow( 1, 1, 28 );
  l_data_abe_no_pad.copy_( at::randn( {28, 28, 64} ) );

  at::Tensor l_data_eab_aten     = l_data_abe_no_pad.permute( {2, 0, 1} ).clone();
  at::Tensor l_data_fab_aux_aten = l_data_abf_aux.permute( {2, 0, 1} );
  at::Tensor l_data_iab_aux_aten = l_data_abi_aux.permute( {2, 0, 1} );

  // evaluate
  l_node_abi.eval();

  // compute reference of first binary contraction
  at::Tensor l_data_fab_aten = at::conv2d( l_data_eab_aten,
                                           l_data_fecd,
                                           {},
                                           1,
                                           1 );

  l_data_fab_aten += l_data_fab_aux_aten;
  l_data_fab_aten = at::relu( l_data_fab_aten );

  // compare result of first binary contraction
  at::Tensor l_data_fab = l_data_abf.permute( {2, 0, 1} );
  l_data_fab = l_data_fab.narrow( 1, 1, 28 ).narrow( 2, 1, 28 );

  REQUIRE( at::allclose( l_data_fab_aten,
                         l_data_fab,
                         1E-3,
                         1E-4 ) );

  l_data_eab_aten += at::conv2d( l_data_fab_aten,
                                 l_data_ifgh,
                                 {},
                                 1,
                                 1 );
  l_data_eab_aten += l_data_iab_aux_aten;

  REQUIRE( at::allclose( l_data_eab_aten,
                         l_data_abe_no_pad.permute( {2, 0, 1} ),
                         1E-2,
                         1E-3 ) );
}
TEST_CASE( "Einsum expression performing ResNet-18 block with downsampling.", "[einsum_node]" ) {
  // layer2 of torchvision.models.resnet18( weights = "ResNet18_Weights.DEFAULT" )
  //
  // (layer2): Sequential(
  //   (0): BasicBlock(
  //     (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  //     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //     (relu): ReLU(inplace=True)
  //     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  //     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //     (downsample): Sequential(
  //       (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
  //       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //     )
  //   )
  //   (1): BasicBlock(
  //     (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  //     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //     (relu): ReLU(inplace=True)
  //     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  //     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  //   )
  // )

  // corresponding einsum-like tree:
  //
  //                       | 
  //               ________+_______
  //              /                 \
  //             /               ___abi___
  //            /              /           \
  //       ___abl___     ___abf___         ifgh
  //     /           \ /           \
  //  lejk           abe          fecd
  //
  std::map< int64_t, int64_t > l_dim_sizes_inner;
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'a',   14 ) ); // height
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'b',   14 ) ); // width
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'c',    3 ) ); // first kernel (height)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'd',    3 ) ); // first kernel (width)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'e',   64 ) ); // input features
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'f',  128 ) ); // intermediate features
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'g',    3 ) ); // second kernel (height)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'h',    3 ) ); // second kernel (width)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'i',  128 ) ); // output features
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'j',    3 ) ); // downsampling kernel (height)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'k',    3 ) ); // downsampling kernel (width)
  l_dim_sizes_inner.insert( std::pair< int64_t, int64_t >( 'l',  128 ) ); // output features downsampling

  std::map< int64_t, int64_t > l_dim_sizes_outer_input;
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'a', 28+2 ) ); // height
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'b', 28+2 ) ); // width
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'c',    3 ) ); // first kernel (height)
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'd',    3 ) ); // first kernel (width)
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'e',   64 ) ); // input features
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'f',  128 ) ); // intermediate features
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'j',    3 ) ); // downsampling kernel (height)
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'k',    3 ) ); // downsampling kernel (width)
  l_dim_sizes_outer_input.insert( std::pair< int64_t, int64_t >( 'l',  128 ) ); // output features downsampling

  std::map< int64_t, int64_t > l_dim_sizes_outer_output;
  l_dim_sizes_outer_output.insert( std::pair< int64_t, int64_t >( 'a', 14+2 ) ); // height
  l_dim_sizes_outer_output.insert( std::pair< int64_t, int64_t >( 'b', 14+2 ) ); // width
  l_dim_sizes_outer_output.insert( std::pair< int64_t, int64_t >( 'f',  128 ) ); // intermediate features
  l_dim_sizes_outer_output.insert( std::pair< int64_t, int64_t >( 'g',    3 ) ); // second kernel (height)
  l_dim_sizes_outer_output.insert( std::pair< int64_t, int64_t >( 'h',    3 ) ); // second kernel (width)
  l_dim_sizes_outer_output.insert( std::pair< int64_t, int64_t >( 'i',  128 ) ); // output features
  l_dim_sizes_outer_output.insert( std::pair< int64_t, int64_t >( 'l',  128 ) ); // output features downsampling

  std::map< int64_t, int64_t > l_dim_sizes_aux_outer;
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'a',   1 ) ); // height
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'b',   1 ) ); // width
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'f', 128 ) ); // intermediate features
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'i', 128 ) ); // otuput features
  l_dim_sizes_aux_outer.insert( std::pair< int64_t, int64_t >( 'l', 128 ) ); // output features downsampling

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'c', 'a' ) ); // first convolution, height
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'd', 'b' ) ); // first convolution, width
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'g', 'a' ) ); // second convolution, height
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'h', 'b' ) ); // second convolution, width
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'j', 'a' ) ); // downsampling convolution, height
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 'k', 'b' ) ); // downsampling convolution, width

  std::map< int64_t, int64_t > l_offsets;
  l_offsets.insert( std::pair< int64_t, int64_t >( 'a',  1 ) );
  l_offsets.insert( std::pair< int64_t, int64_t >( 'b',  1 ) );

  int64_t l_dim_ids_abe[3]  = { 'a', 'b', 'e' };
  int64_t l_dim_ids_fecd[4] = { 'f', 'e', 'c', 'd' };
  int64_t l_dim_ids_abf[3]  = { 'a', 'b', 'f' };
  int64_t l_dim_ids_ifgh[4] = { 'i', 'f', 'g', 'h' };
  int64_t l_dim_ids_abi[3]  = { 'a', 'b', 'i' };
  int64_t l_dim_ids_abl[3]  = { 'a', 'b', 'l' };
  int64_t l_dim_ids_lejk[4] = { 'l', 'e', 'j', 'k' };

  std::vector< int64_t > l_sizes_abe;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abe.push_back( l_dim_sizes_outer_input.at( l_dim_ids_abe[l_di] ) );
  }

  std::vector< int64_t > l_sizes_fecd;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_fecd.push_back( l_dim_sizes_outer_input.at( l_dim_ids_fecd[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abf;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abf.push_back( l_dim_sizes_outer_output.at( l_dim_ids_abf[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abf_aux;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abf_aux.push_back( l_dim_sizes_aux_outer.at( l_dim_ids_abf[l_di] ) );
  }

  std::vector< int64_t > l_sizes_ifgh;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_ifgh.push_back( l_dim_sizes_outer_output.at( l_dim_ids_ifgh[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abi_aux;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abi_aux.push_back( l_dim_sizes_aux_outer.at( l_dim_ids_abi[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abl;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abl.push_back( l_dim_sizes_outer_output.at( l_dim_ids_abl[l_di] ) );
  }

  std::vector< int64_t > l_sizes_lejk;
  for( int64_t l_di = 0; l_di < 4; l_di++ ) {
    l_sizes_lejk.push_back( l_dim_sizes_outer_input.at( l_dim_ids_lejk[l_di] ) );
  }

  std::vector< int64_t > l_sizes_abl_aux;
  for( int64_t l_di = 0; l_di < 3; l_di++ ) {
    l_sizes_abl_aux.push_back( l_dim_sizes_aux_outer.at( l_dim_ids_abl[l_di] ) );
  }

  at::Tensor l_data_abe     = at::zeros( l_sizes_abe    );
  at::Tensor l_data_fecd    = at::randn( l_sizes_fecd    );
  at::Tensor l_data_abf_aux = at::randn( l_sizes_abf_aux );
  at::Tensor l_data_abf     = at::zeros( l_sizes_abf    );
  at::Tensor l_data_ifgh    = at::randn( l_sizes_ifgh    );
  at::Tensor l_data_abi_aux = at::randn( l_sizes_abi_aux );
  at::Tensor l_data_abl     = at::zeros( l_sizes_abl     );
  at::Tensor l_data_lejk    = at::randn( l_sizes_lejk     );
  at::Tensor l_data_abl_aux = at::randn( l_sizes_abl_aux );

  std::map< int64_t, int64_t > l_strides_abe;
  l_strides_abe.insert( std::pair< int64_t, int64_t >( 'a', 2 ) );
  l_strides_abe.insert( std::pair< int64_t, int64_t >( 'b', 2 ) );

  einsum_ir::backend::EinsumNode l_node_abe;
  einsum_ir::backend::EinsumNode l_node_abe_downsampling;
  einsum_ir::backend::EinsumNode l_node_fecd;
  einsum_ir::backend::EinsumNode l_node_abf;
  einsum_ir::backend::EinsumNode l_node_ifgh;
  einsum_ir::backend::EinsumNode l_node_abi;
  einsum_ir::backend::EinsumNode l_node_abl;
  einsum_ir::backend::EinsumNode l_node_lejk;

  // leaf nodes
  l_node_abe.init( 3,
                   l_dim_ids_abe,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_outer_input,
                   einsum_ir::FP32,
                   l_data_abe.data_ptr() );

  l_node_abe_downsampling.init( 3,
                                l_dim_ids_abe,
                                &l_dim_sizes_inner,
                                &l_dim_sizes_outer_input,
                                einsum_ir::FP32,
                                l_data_abe.data_ptr() );

  l_node_fecd.init( 4,
                    l_dim_ids_fecd,
                    &l_dim_sizes_inner,
                    &l_dim_sizes_outer_input,
                    einsum_ir::FP32,
                    l_data_fecd.data_ptr() );

  l_node_ifgh.init( 4,
                    l_dim_ids_ifgh,
                    &l_dim_sizes_inner,
                    &l_dim_sizes_outer_output,
                    einsum_ir::FP32,
                    l_data_ifgh.data_ptr() );

  l_node_lejk.init( 4,
                    l_dim_ids_lejk,
                    &l_dim_sizes_inner,
                    &l_dim_sizes_outer_input,
                    einsum_ir::FP32,
                    l_data_lejk.data_ptr() );

  // dependent nodes
  l_node_abl.init( 3,
                   l_dim_ids_abl,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_aux_outer,
                   &l_dim_sizes_outer_output,
                   nullptr,
                   &l_offsets,
                   &l_strides_abe,
                   nullptr,
                   nullptr,
                   &l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   l_data_abl_aux.data_ptr(),
                   l_data_abl.data_ptr(),
                   einsum_ir::kernel_t::COPY,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::RELU,
                   &l_node_abe_downsampling,
                   &l_node_lejk );

  l_node_abf.init( 3,
                   l_dim_ids_abf,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_aux_outer,
                   &l_dim_sizes_outer_output,
                   nullptr,
                   &l_offsets,
                   &l_strides_abe,
                   nullptr,
                   nullptr,
                   &l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   l_data_abf_aux.data_ptr(),
                   l_data_abf.data_ptr(),
                   einsum_ir::kernel_t::COPY,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::RELU,
                   &l_node_abe,
                   &l_node_fecd );

  l_node_abi.init( 3,
                   l_dim_ids_abi,
                   &l_dim_sizes_inner,
                   &l_dim_sizes_aux_outer,
                   &l_dim_sizes_outer_output,
                   nullptr,
                   &l_offsets,
                   nullptr,
                   nullptr,
                   nullptr,
                   &l_dim_link_s_to_p,
                   einsum_ir::FP32,
                   l_data_abi_aux.data_ptr(),
                   l_data_abl.data_ptr(),
                   einsum_ir::kernel_t::ADD,
                   einsum_ir::kernel_t::MADD,
                   einsum_ir::kernel_t::UNDEFINED_KTYPE,
                   &l_node_abf,
                   &l_node_ifgh );

  // compile
  einsum_ir::err_t l_err = einsum_ir::err_t::UNDEFINED_ERROR;
  l_err = l_node_abl.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );
  l_err = l_node_abi.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // copy in the filters
  l_node_fecd.store_and_lock_data();
  l_node_ifgh.store_and_lock_data();
  l_node_lejk.store_and_lock_data();

  // assign input data
  at::Tensor l_data_abe_no_pad = l_data_abe.narrow( 0, 1, 28 );
  l_data_abe_no_pad = l_data_abe_no_pad.narrow( 1, 1, 28 );
  l_data_abe_no_pad.copy_( at::randn( {28, 28, 64} ) );

  at::Tensor l_data_eab_aten     = l_data_abe_no_pad.permute( {2, 0, 1} ).clone();
  at::Tensor l_data_fab_aux_aten = l_data_abf_aux.permute( {2, 0, 1} );
  at::Tensor l_data_iab_aux_aten = l_data_abi_aux.permute( {2, 0, 1} );
  at::Tensor l_data_lab_aux_aten = l_data_abl_aux.permute( {2, 0, 1} );

  // eval downsampling
  l_node_abl.eval();

  // compute reference of downsampling
  at::Tensor l_data_lab_aten = at::conv2d( l_data_eab_aten,
                                           l_data_lejk,
                                           {},
                                           2,
                                           1 );

  l_data_lab_aten += l_data_lab_aux_aten;
  l_data_lab_aten = at::relu( l_data_lab_aten );

  // compare result of downsampling
  at::Tensor l_data_lab = l_data_abl.permute( {2, 0, 1} );
  l_data_lab = l_data_lab.narrow( 1, 1, 14 ).narrow( 2, 1, 14 );

  REQUIRE( at::allclose( l_data_lab_aten,
                         l_data_lab,
                         1E-3,
                         1E-4 ) );

  // evaluate
  l_node_abi.eval();

  // compute reference of first binary contraction
  at::Tensor l_data_fab_aten = at::conv2d( l_data_eab_aten,
                                           l_data_fecd,
                                           {},
                                           2,
                                           1 );

  l_data_fab_aten += l_data_fab_aux_aten;
  l_data_fab_aten = at::relu( l_data_fab_aten );

  // compare result of first main binary contraction
  at::Tensor l_data_fab = l_data_abf.permute( {2, 0, 1} );
  l_data_fab = l_data_fab.narrow( 1, 1, 14 ).narrow( 2, 1, 14 );

  REQUIRE( at::allclose( l_data_fab_aten,
                         l_data_fab,
                         1E-3,
                         1E-4 ) );

  // compute reference of second main contraction (added to abl result)
  l_data_lab_aten += at::conv2d( l_data_fab_aten,
                                 l_data_ifgh,
                                 {},
                                 1,
                                 1 );
  l_data_lab_aten += l_data_iab_aux_aten;

  // unpad output
  at::Tensor l_data_abl_no_pad = l_data_abl.narrow( 0, 1, 14 );
  l_data_abl_no_pad = l_data_abl_no_pad.narrow( 1, 1, 14 );

  REQUIRE( at::allclose( l_data_lab_aten,
                         l_data_abl_no_pad.permute( {2, 0, 1} ),
                         1E-2,
                         1E-3 ) );
}