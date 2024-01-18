#include "catch.hpp"
#include "BinaryContraction.h"

TEST_CASE( "Derives the dimension types given a binary einsum string of int64 values", "[bin_cont_dim_types]" ) {
  //         ______________iexfghy________________
  //        /                                     \
  //   ie-a-x-c-g-y                            a-f-x-c-h-y
  //
  // i: 0
  // e: 1
  // x: 2
  // f: 3
  // g: 4
  // h: 5
  // y: 6
  // a: 7
  // c: 8
  int64_t l_ids_out[7] = { 0, 1, 2, 3, 4, 5, 6 };
  int64_t l_ids_in_left[7] = { 0, 1, 7, 2, 8, 4, 6 };
  int64_t l_ids_in_right[6] = { 7, 3, 2, 8, 5, 6 };

  einsum_ir::dim_t l_types_out[7] = { einsum_ir::UNDEFINED_DIM };
  einsum_ir::dim_t l_types_in_left[7] = { einsum_ir::UNDEFINED_DIM };
  einsum_ir::dim_t l_types_in_right[6] = { einsum_ir::UNDEFINED_DIM };

  einsum_ir::backend::BinaryContraction::dim_types( 7,
                                                    6,
                                                    7,
                                                    l_ids_in_left,
                                                    l_ids_in_right,
                                                    l_ids_out,
                                                    einsum_ir::UNDEFINED_DIM,
                                                    einsum_ir::M,
                                                    einsum_ir::N,
                                                    einsum_ir::C,
                                                    l_types_out );

  REQUIRE( l_types_out[0] == einsum_ir::M );
  REQUIRE( l_types_out[1] == einsum_ir::M );
  REQUIRE( l_types_out[2] == einsum_ir::C );
  REQUIRE( l_types_out[3] == einsum_ir::N );
  REQUIRE( l_types_out[4] == einsum_ir::M );
  REQUIRE( l_types_out[5] == einsum_ir::N );
  REQUIRE( l_types_out[6] == einsum_ir::C );

  einsum_ir::backend::BinaryContraction::dim_types( 6,
                                                    7,
                                                    7,
                                                    l_ids_in_right,
                                                    l_ids_out,
                                                    l_ids_in_left,
                                                    einsum_ir::I,
                                                    einsum_ir::K,
                                                    einsum_ir::M,
                                                    einsum_ir::C,
                                                    l_types_in_left );

  REQUIRE( l_types_in_left[0] == einsum_ir::M );
  REQUIRE( l_types_in_left[1] == einsum_ir::M );
  REQUIRE( l_types_in_left[2] == einsum_ir::K );
  REQUIRE( l_types_in_left[3] == einsum_ir::C );
  REQUIRE( l_types_in_left[4] == einsum_ir::K );
  REQUIRE( l_types_in_left[5] == einsum_ir::M );
  REQUIRE( l_types_in_left[6] == einsum_ir::C );

  einsum_ir::backend::BinaryContraction::dim_types( 7,
                                                    7,
                                                    6,
                                                    l_ids_in_left,
                                                    l_ids_out,
                                                    l_ids_in_right,
                                                    einsum_ir::J,
                                                    einsum_ir::K,
                                                    einsum_ir::N,
                                                    einsum_ir::C,
                                                    l_types_in_right );

  REQUIRE( l_types_in_right[0] == einsum_ir::K );
  REQUIRE( l_types_in_right[1] == einsum_ir::N );
  REQUIRE( l_types_in_right[2] == einsum_ir::C );
  REQUIRE( l_types_in_right[3] == einsum_ir::K );
  REQUIRE( l_types_in_right[4] == einsum_ir::N );
  REQUIRE( l_types_in_right[5] == einsum_ir::C );
}

TEST_CASE( "Filters the dimension ids based on the given type.", "[filter_dim_ids]" ) {
  int64_t l_ids_out[7] = { 10, 11, 12, 13, 14, 15, 16 };
  einsum_ir::dim_t l_types_out[7] = { einsum_ir::M,   // 10
                                      einsum_ir::M,   // 11
                                      einsum_ir::C,   // 12
                                      einsum_ir::N,   // 13
                                      einsum_ir::M,   // 14
                                      einsum_ir::N,   // 15
                                      einsum_ir::C }; // 16

  int64_t l_ids_m[3] = { 0 };
  int64_t l_ids_n[2] = { 0 };
  int64_t l_ids_c[2] = { 0 };

  int64_t l_num_dims_m = einsum_ir::backend::BinaryContraction::filter_dim_ids( 7,
                                                                                einsum_ir::M,
                                                                                l_types_out,
                                                                                l_ids_out,
                                                                                l_ids_m );
  REQUIRE( l_num_dims_m == 3 );
  REQUIRE( l_ids_m[0] == 10 );
  REQUIRE( l_ids_m[1] == 11 );
  REQUIRE( l_ids_m[2] == 14 );

  int64_t l_num_dims_n = einsum_ir::backend::BinaryContraction::filter_dim_ids( 7,
                                                                                einsum_ir::N,
                                                                                l_types_out,
                                                                                l_ids_out,
                                                                                l_ids_n );
  REQUIRE( l_num_dims_n == 2 );
  REQUIRE( l_ids_n[0] == 13 );
  REQUIRE( l_ids_n[1] == 15 );

  int64_t l_num_dims_c = einsum_ir::backend::BinaryContraction::filter_dim_ids( 7,
                                                                                einsum_ir::C,
                                                                                l_types_out,
                                                                                l_ids_out,
                                                                                l_ids_c );
  REQUIRE( l_num_dims_c == 2 );
  REQUIRE( l_ids_c[0] == 12 );
  REQUIRE( l_ids_c[1] == 16 );
}

TEST_CASE( "Orders the dimensions of the inputs (left_bc_bm_bk_right_bc_bn_bk_out_native)", "[bin_cont_order_dims_in]" ) {
  int64_t l_ids_c[2] = { 0, 1 };
  int64_t l_ids_m[3] = { 2, 3, 4 };
  int64_t l_ids_n[2] = { 5, 6 };
  int64_t l_ids_k[2] = { 7, 8 };

  int64_t l_dim_ids_left[7] = { 0 };
  int64_t l_dim_ids_right[6] = { 0 };

  einsum_ir::err_t l_err = einsum_ir::backend::BinaryContraction::order_dims_in( einsum_ir::LEFT_BC_BM_BI_BK_RIGHT_BC_BN_BJ_BK_OUT_NATIVE,
                                                                                 2,
                                                                                 3,
                                                                                 2,
                                                                                 2,
                                                                                 0,
                                                                                 0,
                                                                                 0,
                                                                                 0,
                                                                                 0,
                                                                                 0,
                                                                                 0,
                                                                                 0,
                                                                                 l_ids_c,
                                                                                 l_ids_m,
                                                                                 l_ids_n,
                                                                                 l_ids_k,
                                                                                 nullptr,
                                                                                 nullptr,
                                                                                 l_dim_ids_left,
                                                                                 l_dim_ids_right );

  REQUIRE( l_err == einsum_ir::SUCCESS );

  REQUIRE( l_dim_ids_left[6] == 8 );
  REQUIRE( l_dim_ids_left[5] == 7 );
  REQUIRE( l_dim_ids_left[4] == 4 );
  REQUIRE( l_dim_ids_left[3] == 3 );
  REQUIRE( l_dim_ids_left[2] == 2 );
  REQUIRE( l_dim_ids_left[1] == 1 );
  REQUIRE( l_dim_ids_left[0] == 0 );

  REQUIRE( l_dim_ids_right[5] == 8 );
  REQUIRE( l_dim_ids_right[4] == 7 );
  REQUIRE( l_dim_ids_right[3] == 6 );
  REQUIRE( l_dim_ids_right[2] == 5 );
  REQUIRE( l_dim_ids_right[1] == 1 );
  REQUIRE( l_dim_ids_right[0] == 0 );
}

TEST_CASE( "Orders the dimensions of the inputs (left_bc_bm_bk_kb_mb_right_bc_bn_bk_nb_kb_out_native).", "[bin_cont_order_dims_in]" ) {
  int64_t l_ids_c[2] = { 0, 1 };
  int64_t l_ids_m[3] = { 2, 3, 4 };
  int64_t l_ids_n[2] = { 5, 6 };
  int64_t l_ids_k[2] = { 7, 8 };

  int64_t l_dim_ids_left[7] = { 0 };
  int64_t l_dim_ids_right[6] = { 0 };

  einsum_ir::err_t l_err = einsum_ir::backend::BinaryContraction::order_dims_in( einsum_ir::LEFT_BC_BM_BI_BK_IB_KB_MB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_OUT_NATIVE,
                                                                                 2,
                                                                                 3,
                                                                                 2,
                                                                                 2,
                                                                                 0,
                                                                                 0,
                                                                                 0,
                                                                                 1,
                                                                                 1,
                                                                                 1,
                                                                                 0,
                                                                                 0,
                                                                                 l_ids_c,
                                                                                 l_ids_m,
                                                                                 l_ids_n,
                                                                                 l_ids_k,
                                                                                 nullptr,
                                                                                 nullptr,
                                                                                 l_dim_ids_left,
                                                                                 l_dim_ids_right );

  REQUIRE( l_err == einsum_ir::SUCCESS );

  REQUIRE( l_dim_ids_left[6] == 4 );
  REQUIRE( l_dim_ids_left[5] == 8 );
  REQUIRE( l_dim_ids_left[4] == 7 );
  REQUIRE( l_dim_ids_left[3] == 3 );
  REQUIRE( l_dim_ids_left[2] == 2 );
  REQUIRE( l_dim_ids_left[1] == 1 );
  REQUIRE( l_dim_ids_left[0] == 0 );

  REQUIRE( l_dim_ids_right[5] == 8 );
  REQUIRE( l_dim_ids_right[4] == 6 );
  REQUIRE( l_dim_ids_right[3] == 7 );
  REQUIRE( l_dim_ids_right[2] == 5 );
  REQUIRE( l_dim_ids_right[1] == 1 );
  REQUIRE( l_dim_ids_right[0] == 0 );

  l_err = einsum_ir::backend::BinaryContraction::order_dims_in( einsum_ir::LEFT_BC_BM_BI_BK_IB_KB_MB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_OUT_NATIVE,
                                                                2,
                                                                3,
                                                                2,
                                                                2,
                                                                0,
                                                                0,
                                                                0,
                                                                2,
                                                                1,
                                                                1,
                                                                0,
                                                                0,
                                                                l_ids_c,
                                                                l_ids_m,
                                                                l_ids_n,
                                                                l_ids_k,
                                                                nullptr,
                                                                nullptr,
                                                                l_dim_ids_left,
                                                                l_dim_ids_right );

  REQUIRE( l_err == einsum_ir::SUCCESS );

  REQUIRE( l_dim_ids_left[6] == 4 );
  REQUIRE( l_dim_ids_left[5] == 3 );
  REQUIRE( l_dim_ids_left[4] == 8 );
  REQUIRE( l_dim_ids_left[3] == 7 );
  REQUIRE( l_dim_ids_left[2] == 2 );
  REQUIRE( l_dim_ids_left[1] == 1 );
  REQUIRE( l_dim_ids_left[0] == 0 );

  REQUIRE( l_dim_ids_right[5] == 8 );
  REQUIRE( l_dim_ids_right[4] == 6 );
  REQUIRE( l_dim_ids_right[3] == 7 );
  REQUIRE( l_dim_ids_right[2] == 5 );
  REQUIRE( l_dim_ids_right[1] == 1 );
  REQUIRE( l_dim_ids_right[0] == 0 );
}

TEST_CASE( "Orders the dimensions of the inputs (left_bc_bm_bk_kb_mb_cb_right_bc_bn_bk_nb_kb_cb_out_native).", "[bin_cont_order_dims_in]" ) {
  int64_t l_ids_c[3] = { 0, 1, 2 };
  int64_t l_ids_m[3] = { 3, 4, 5 };
  int64_t l_ids_n[2] = { 6, 7 };
  int64_t l_ids_k[2] = { 8, 9 };

  int64_t l_dim_ids_left[8] = { 0 };
  int64_t l_dim_ids_right[7] = { 0 };

  einsum_ir::err_t l_err = einsum_ir::backend::BinaryContraction::order_dims_in( einsum_ir::LEFT_BC_BM_BI_BK_IB_KB_MB_CB_RIGHT_BC_BN_BJ_BK_NB_JB_KB_CB_OUT_NATIVE,
                                                                                  3,
                                                                                  3,
                                                                                  2,
                                                                                  2,
                                                                                  0,
                                                                                  0,
                                                                                  2,
                                                                                  1,
                                                                                  2,
                                                                                  1,
                                                                                  0,
                                                                                  0,
                                                                                  l_ids_c,
                                                                                  l_ids_m,
                                                                                  l_ids_n,
                                                                                  l_ids_k,
                                                                                  nullptr,
                                                                                  nullptr,
                                                                                  l_dim_ids_left,
                                                                                  l_dim_ids_right );

  REQUIRE( l_err == einsum_ir::SUCCESS );

  REQUIRE( l_dim_ids_left[7] == 2 );
  REQUIRE( l_dim_ids_left[6] == 1 );
  REQUIRE( l_dim_ids_left[5] == 5 );
  REQUIRE( l_dim_ids_left[4] == 9 );
  REQUIRE( l_dim_ids_left[3] == 8 );
  REQUIRE( l_dim_ids_left[2] == 4 );
  REQUIRE( l_dim_ids_left[1] == 3 );
  REQUIRE( l_dim_ids_left[0] == 0 );

  REQUIRE( l_dim_ids_right[6] == 2 );
  REQUIRE( l_dim_ids_right[5] == 1 );
  REQUIRE( l_dim_ids_right[4] == 9 );
  REQUIRE( l_dim_ids_right[3] == 7 );
  REQUIRE( l_dim_ids_right[2] == 6 );
  REQUIRE( l_dim_ids_right[1] == 8 );
  REQUIRE( l_dim_ids_right[0] == 0 );
}

TEST_CASE( "Orders the dimensions of the inputs (left_bc_bm_bi_bk_cb_kb_mb_right_bc_bn_bj_bk_cb_nb_kb_out_native).", "[bin_cont_order_dims_in]" ) {
  int64_t l_ids_c[3] = { 0, 1, 2 };
  int64_t l_ids_m[3] = { 3, 4, 5 };
  int64_t l_ids_n[2] = { 6, 7 };
  int64_t l_ids_k[2] = { 8, 9 };

  int64_t l_dim_ids_left[8] = { 0 };
  int64_t l_dim_ids_right[7] = { 0 };

  einsum_ir::err_t l_err = einsum_ir::backend::BinaryContraction::order_dims_in( einsum_ir::LEFT_BC_BM_BI_BK_CB_KB_MB_RIGHT_BC_BN_BJ_BK_CB_NB_KB_OUT_NATIVE,
                                                                                 3,
                                                                                 3,
                                                                                 2,
                                                                                 2,
                                                                                 0,
                                                                                 0,
                                                                                 2, // cb
                                                                                 1, // mb
                                                                                 2, // nb
                                                                                 1, // kb
                                                                                 0, // ib
                                                                                 0, // jb
                                                                                 l_ids_c,
                                                                                 l_ids_m,
                                                                                 l_ids_n,
                                                                                 l_ids_k,
                                                                                 nullptr,
                                                                                 nullptr,
                                                                                 l_dim_ids_left,
                                                                                 l_dim_ids_right );

  REQUIRE( l_err == einsum_ir::SUCCESS );

  REQUIRE( l_dim_ids_left[7] == 5 );
  REQUIRE( l_dim_ids_left[6] == 9 );
  REQUIRE( l_dim_ids_left[5] == 2 );
  REQUIRE( l_dim_ids_left[4] == 1 );
  REQUIRE( l_dim_ids_left[3] == 8 );
  REQUIRE( l_dim_ids_left[2] == 4 );
  REQUIRE( l_dim_ids_left[1] == 3 );
  REQUIRE( l_dim_ids_left[0] == 0 );

  REQUIRE( l_dim_ids_right[6] == 9 );
  REQUIRE( l_dim_ids_right[5] == 7 );
  REQUIRE( l_dim_ids_right[4] == 6 );
  REQUIRE( l_dim_ids_right[3] == 2 );
  REQUIRE( l_dim_ids_right[2] == 1 );
  REQUIRE( l_dim_ids_right[1] == 8 );
  REQUIRE( l_dim_ids_right[0] == 0 );
}

TEST_CASE( "Links secondary to primary dimensions w.r.t. to a primary tensor.", "[bin_cont_link]" ) {
  int64_t l_dim_id_s = 4;
  int64_t l_num_dims_p = 5;
  int64_t l_dim_ids_p[5] = { 2, 5, 1, 3, 7 };

  std::map< int64_t, int64_t > l_dim_link_s_to_p;
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 10,  2 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 11, 12 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >(  4,  3 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 13, 14 ) );
  l_dim_link_s_to_p.insert( std::pair< int64_t, int64_t >( 14,  2 ) );

  int64_t l_loc = 0;
  einsum_ir::err_t l_err = einsum_ir::backend::BinaryContraction::link_secondary_to_primary( l_dim_id_s,
                                                                                             l_num_dims_p,
                                                                                             l_dim_ids_p,
                                                                                             l_dim_link_s_to_p,
                                                                                             l_loc );
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );
  REQUIRE( l_loc == 3 );
}