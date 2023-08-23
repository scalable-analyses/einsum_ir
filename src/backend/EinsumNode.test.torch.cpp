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
                 l_dim_sizes,
                 einsum_ir::FP32,
                 l_in_left.data_ptr() );

  l_node_1.init( 2,
                 l_dim_ids_in_right,
                 l_dim_sizes,
                 einsum_ir::FP32,
                 l_in_right.data_ptr() );

  l_node_2.init( 2,
                 l_dim_ids_out,
                 einsum_ir::FP32,
                 l_out.data_ptr(),
                 einsum_ir::UNDEFINED_KTYPE,
                 einsum_ir::MADD,
                 einsum_ir::UNDEFINED_KTYPE,
                 l_node_0,
                 l_node_1 );

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
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_ca.data_ptr() );

  l_node_bc.init( 2,
                  l_dim_ids_bc,
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_bc.data_ptr() );

  l_node_da.init( 2,
                  l_dim_ids_da,
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_da.data_ptr() );

  l_node_ba.init( 2,
                  l_dim_ids_ba,
                  einsum_ir::FP32,
                  l_data_ba.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  l_node_ca,
                  l_node_bc );

  l_node_bd.init( 2,
                  l_dim_ids_bd,
                  einsum_ir::FP32,
                  l_data_bd.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  l_node_ba,
                  l_node_da );

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
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_ca.data_ptr() );

  l_node_bc.init( 2,
                  l_dim_ids_bc,
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_bc.data_ptr() );

  l_node_da.init( 2,
                  l_dim_ids_da,
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_da.data_ptr() );

  l_node_ba.init( 2,
                  l_dim_ids_ba,
                  einsum_ir::FP32,
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  l_node_ca,
                  l_node_bc );

  l_node_bd.init( 2,
                  l_dim_ids_bd,
                  einsum_ir::FP32,
                  l_data_bd.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  l_node_ba,
                  l_node_da );

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
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_mk.data_ptr() );

  l_node_kn.init( 2,
                  l_dim_ids_kn,
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_kn.data_ptr() );

  l_node_nm.init( 2,
                  l_dim_ids_nm,
                  einsum_ir::FP32,
                  l_data_nm.data_ptr(),
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE,
                  l_node_mk,
                  l_node_kn );

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
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_hd.data_ptr() );

  l_node_fb.init( 2,
                  l_dim_ids_fb,
                  l_dim_sizes,
                  einsum_ir::FP32,
                  l_data_fb.data_ptr() );

  l_node_abcd.init( 4,
                    l_dim_ids_abcd,
                    l_dim_sizes,
                    einsum_ir::FP32,
                    l_data_abcd.data_ptr() );

  l_node_eai.init( 3,
                   l_dim_ids_eai,
                   l_dim_sizes,
                   einsum_ir::FP32,
                   l_data_eai.data_ptr() );

  l_node_gic.init( 3,
                   l_dim_ids_gic,
                   l_dim_sizes,
                   einsum_ir::FP32,
                   l_data_gic.data_ptr() );

  // dependent nodes
  l_node_hacf.init( 4,
                    l_dim_ids_hacf,
                    einsum_ir::FP32,
                    einsum_ir::ZERO,
                    einsum_ir::MADD,
                    einsum_ir::UNDEFINED_KTYPE,
                    l_node_hd,
                    l_node_facd );

  l_node_facd.init( 4,
                    l_dim_ids_facd,
                    einsum_ir::FP32,
                    einsum_ir::ZERO,
                    einsum_ir::MADD,
                    einsum_ir::UNDEFINED_KTYPE,
                    l_node_fb,
                    l_node_abcd );

  l_node_iaecg.init( 5,
                     l_dim_ids_iaecg,
                     einsum_ir::FP32,
                     einsum_ir::ZERO,
                     einsum_ir::MADD,
                     einsum_ir::UNDEFINED_KTYPE,
                     l_node_eai,
                     l_node_gic );

  l_node_iefgh.init( 5,
                     l_dim_ids_iefgh,
                     einsum_ir::FP32,
                     l_data_iefgh.data_ptr(),
                     einsum_ir::ZERO,
                     einsum_ir::MADD,
                     einsum_ir::UNDEFINED_KTYPE,
                     l_node_hacf,
                     l_node_iaecg );

  l_node_iefgh.compile();
  l_node_iefgh.eval();

  // reference
  l_data_iefgh_ref = at::einsum( "hd,fb,abcd,eai,gic->iefgh",
                                 {l_data_hd, l_data_fb, l_data_abcd, l_data_eai, l_data_gic} );


  REQUIRE( at::allclose( l_data_iefgh_ref, l_data_iefgh ) );
}