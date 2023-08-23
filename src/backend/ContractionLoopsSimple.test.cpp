#include "catch.hpp"
#include "ContractionLoopsSimple.h"
#include <cmath>

/**
 * Generic matrix multiplication C += AB.
 *
 * @param i_a input matrix A.
 * @param i_b input matrix B.
 * @param io_c matrix C.
 * @param i_m BLAS parameter M.
 * @param i_n BLAS parameter N.
 * @param i_k BLAS parameter K.
 * @param i_lda leading dimension of A.
 * @param i_ldb leading dimension of B.
 * @param i_ldc leading dimension of C.
 **/
void gemm_ref_mnk( float const * i_a,
                   float const * i_b,
                   float       * io_c,
                   int64_t       i_m,
                   int64_t       i_n,
                   int64_t       i_k,
                   int64_t       i_lda,
                   int64_t       i_ldb,
                   int64_t       i_ldc ) {
  for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
    for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
      for( int64_t l_k = 0; l_k < i_k; l_k++ ) {
        io_c[ l_n*i_ldc + l_m ] += i_a[ l_k*i_lda + l_m ] * i_b[ l_n*i_ldb + l_k ];
      }
    }
  }
}

/**
 * Scalar kernel which computes out += left * right.
 *
 * @param i_in_left left scalar.
 * @param i_in_right right scalar.
 * @param io_out output. 
 **/
void kernel_madd_fp32( void const * i_in_left,
                       void const * i_in_right,
                       void       * io_out ) {
  float const * l_in_left = (float const *) i_in_left;
  float const * l_in_right = (float const *) i_in_right;
  float * l_out = (float *) io_out;

  *l_out += (*l_in_left) * (*l_in_right);
}

/**
 * Scalar zeroing kernel.
 *
 * @param io_data data which will be zeroed.
 **/
void kernel_zero_fp32( void * io_data ) {
  float * l_data = (float *) io_data;
  *l_data = 0;
}

/**
 * Scalar ReLU kernel.
 *
 * @param io_data data to which the ReLU is applied.
 **/
void kenrel_relu_fp32( void * io_data ) {
  float * l_data = (float *) io_data;
  *l_data = std::max( *l_data, 0.0f );
}

/**
 * Matrix multiplication kernel C+=AB with
 *   M, N, K = 24, 5, 2
 *   and
 *   ldA, ldB, ldC = 24, 2 72.
 *
 * @param i_in_left left input matrix (A).
 * @param i_in_right right input matrix (B).
 * @param io_out output matrix (C).
 */
void kernel_mat_24_5_2_24_2_72_fp32( void const * i_in_left,
                                     void const * i_in_right,
                                     void       * io_out ) {
  float const * l_in_left = (float const *) i_in_left;
  float const * l_in_right = (float const *) i_in_right;
  float * l_out = (float *) io_out;

  gemm_ref_mnk( l_in_left,
                l_in_right,
                l_out,
                24,
                5,
                2,
                24,
                2,
                72 );
}

/**
 * Transposes the given float32 tensor.
 *
 * @param i_num_dims number of dimensions.
 * @param i_sizes sizes of the dimensions.
 * @param i_strides_in strides of the input tensor.
 * @param i_strides_out strides of the output tensor.
 * @param i_tensor_in input tensor.
 * @param o_tensor_out output tensor. 
 **/
void transpose( int64_t         i_num_dims,
                int64_t const * i_sizes,
                int64_t const * i_strides_in,
                int64_t const * i_strides_out,
                float   const * i_tensor_in,
                float         * o_tensor_out ) {
  if( i_num_dims > 0 ) {
    float const * l_tensor_in = i_tensor_in;
    float       * l_tensor_out = o_tensor_out;

    for( int64_t l_it = 0; l_it < i_sizes[0]; l_it++ ) {
      transpose( i_num_dims-1,
                 i_sizes+1,
                 i_strides_in+1,
                 i_strides_out+1,
                 l_tensor_in,
                 l_tensor_out );

      l_tensor_in += i_strides_in[0];
      l_tensor_out += i_strides_out[0];
    }
  }
  else {
    *o_tensor_out = *i_tensor_in;
  }
}

TEST_CASE( "K dimension of the contraction loops.", "[contraction_loops_k]" ) {
  // per-dimension sizes
  int64_t l_sizes_c[1] = { 1 }; 
  int64_t l_sizes_m[1] = { 1 };
  int64_t l_sizes_n[1] = { 1 };
  int64_t l_sizes_k[1] = { 5 };

  // strides of the left input tensor
  int64_t l_strides_in_left_c[1] = { 1 };
  int64_t l_strides_in_left_m[1] = { 0 };
  int64_t l_strides_in_left_k[1] = { 1 };

  // strides of the right input tensor
  int64_t l_strides_in_right_c[1] = { 1 };
  int64_t l_strides_in_right_n[1] = { 1 };
  int64_t l_strides_in_right_k[1] = { 1 };

  // strides of the output tensor
  int64_t l_strides_out_c[1] = { 1 };
  int64_t l_strides_out_m[1] = { 1 };
  int64_t l_strides_out_n[1] = { 1 };

  // input tensors
  float l_a[5] = { 1, 2, 3, 4,  5 };
  float l_b[5] = { 6, 7, 8, 9, 10 };
  float l_c[1] = { 0 };

  // reference solution
  float l_ref = 1*6 + 2*7 + 3*8 + 4*9 + 5*10;

  // binary contraction loops
  einsum_ir::backend::ContractionLoopsSimple l_cont_loops;
  l_cont_loops.init( 1,
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
                     l_strides_out_c,
                     l_strides_out_m,
                     l_strides_out_n,
                     4,
                     4,
                     4,
                     nullptr,
                     kernel_madd_fp32,
                     nullptr );

  // single recursion (single loop around kernel)
  l_cont_loops.contract_cnmk( 3,
                              0,
                              l_a,
                              l_b,
                              l_c );
  REQUIRE( l_c[0] == l_ref );

  // two recursions
  l_c[0] = 0;

  l_cont_loops.contract_cnmk(  2,
                               0,
                               l_a,
                               l_b,
                               l_c );
  REQUIRE( l_c[0] == l_ref );

  // three recursions
  l_c[0] = 0;

  l_cont_loops.contract_cnmk(  1,
                               0,
                               l_a,
                               l_b,
                               l_c );
  REQUIRE( l_c[0] == l_ref );

  // four recursions
  l_c[0] = 0;

  l_cont_loops.contract_cnmk(  0,
                               0,
                               l_a,
                               l_b,
                               l_c );
  REQUIRE( l_c[0] == l_ref );

  // wrapper
  l_c[0] = 0;
  l_cont_loops.contract( l_a,
                         l_b,
                         l_c );
  REQUIRE( l_c[0] == l_ref );
}

TEST_CASE( "Matmul with first and last touch.", "[contraction_loops]" ) {
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
  int64_t l_num_dims_c = 1;
  int64_t l_num_dims_m = 1;
  int64_t l_num_dims_n = 1;
  int64_t l_num_dims_k = 1;

  int64_t l_sizes_c[1] = { 1 }; 
  int64_t l_sizes_m[1] = { 5 };
  int64_t l_sizes_n[1] = { 7 };
  int64_t l_sizes_k[1] = { 8 };

  int64_t l_strides_left_c[1] = { 1 };
  int64_t l_strides_left_m[1] = { 1 };
  int64_t l_strides_left_k[1] = { 5 };

  int64_t l_strides_right_c[1] = { 1 };
  int64_t l_strides_right_n[1] = { 8 };
  int64_t l_strides_right_k[1] = { 1 };

  int64_t l_strides_out_c[1] = { 1 };
  int64_t l_strides_out_m[1] = { 1 };
  int64_t l_strides_out_n[1] = { 5 };

  constexpr int64_t l_m = 5;
  constexpr int64_t l_n = 7;
  constexpr int64_t l_k = 8;

  constexpr int64_t l_size_a = l_m*l_k;
  constexpr int64_t l_size_b = l_n*l_k;
  constexpr int64_t l_size_c = l_m*l_n;

  float l_a_mat[l_size_a] = {0};
  float l_b_mat[l_size_b] = {0};
  float l_c_mat[l_size_c] = {0};
  float l_c_mat_ref[l_size_c] = {0};

  // init values
  Catch::Generators::RandomFloatingGenerator< float > l_ran( -1.0f, 1.0f );

  for( int64_t l_en = 0; l_en < l_size_a; l_en++ ) {
    l_a_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  for( int64_t l_en = 0; l_en < l_size_b; l_en++ ) {
    l_b_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  for( int64_t l_en = 0; l_en < l_size_c; l_en++ ) {
    l_c_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  einsum_ir::backend::ContractionLoopsSimple l_cont_loops;
  l_cont_loops.init( l_num_dims_c,
                     l_num_dims_m,
                     l_num_dims_n,
                     l_num_dims_k,
                     l_sizes_c,
                     l_sizes_m,
                     l_sizes_n,
                     l_sizes_k,
                     l_strides_left_c,
                     l_strides_left_m,
                     l_strides_left_k,
                     l_strides_right_c,
                     l_strides_right_n,
                     l_strides_right_k,
                     l_strides_out_c,
                     l_strides_out_m,
                     l_strides_out_n,
                     4,
                     4,
                     4,
                     kernel_zero_fp32,
                     kernel_madd_fp32,
                     kenrel_relu_fp32 );

  l_cont_loops.contract( l_a_mat,
                         l_b_mat,
                         l_c_mat );

  gemm_ref_mnk( l_a_mat,
                l_b_mat,
                l_c_mat_ref,
                l_m,
                l_n,
                l_k,
                l_m,
                l_k,
                l_m );

  for( int64_t l_en = 0; l_en < l_size_c; l_en++ ) {
    l_c_mat_ref[l_en] = std::max( l_c_mat_ref[l_en], 0.0f );
  }

  for( int64_t l_en = 0; l_en < l_size_c; l_en++ ) {
    REQUIRE( l_c_mat[l_en] == Approx( l_c_mat_ref[l_en] ) );
  }
}

TEST_CASE( "Nested loops used in binary contractions using a scalar kernel.", "[contraction_loops_scalar]" ) {
  // Test case:
  //
  //   A: mb-kb-cb-bk-bm-bc
  //   B: kb-nb-cb-bk-bn-bc
  //
  //         ______________iefgh__________________
  //        /                                     \
  //   ie-a--c-g-                              a-f--c-h-
  //
  //   sizes:
  //     i: 3 (m0)
  //     e: 8 (m1)
  //     a: 2 (k0)
  //     c: 7 (k1)
  //     g: 6 (m2)
  //     f: 5 (n0)
  //     h: 4 (n1)
  int64_t l_num_dims_c = 1;
  int64_t l_num_dims_m = 3;
  int64_t l_num_dims_n = 2;
  int64_t l_num_dims_k = 2;

  int64_t l_sizes_c[1] = { 1 }; 
  int64_t l_sizes_m[3] = { 6, 8, 3 };
  int64_t l_sizes_n[2] = { 4, 5 };
  int64_t l_sizes_k[2] = { 7, 2 };


  // in left:
  //   sizes:         6 (g), 7 (c), 2 (a), 8 (e), 3 (i)
  //   strides: 2016,   336,    48,    24,     3,     1
  int64_t l_strides_in_left_c[1] = { 1 };
  int64_t l_strides_in_left_m[3] = { 336, 3, 1 };
  int64_t l_strides_in_left_k[2] = { 48, 24 };

  // in right:
  //   sizes:        4 (h), 7 (c), 5 (f), 2 (a)
  //   strides: 280,    70,    10,     2,     1
  int64_t l_strides_in_right_c[1] = { 1 };
  int64_t l_strides_in_right_n[2] = { 70, 2 };
  int64_t l_strides_in_right_k[2] = { 10, 1 };

  // out:
  //   sizes:           4 (h), 6 (g), 5 (f), 8 (e), 3 (i)
  //   strides:   2880,   720,   120,    24,     3,     1
  int64_t l_strides_out_c[1] = { 1 };
  int64_t l_strides_out_m[3] = { 120, 3, 1 };
  int64_t l_strides_out_n[2] = { 720, 24 };

  // column-major matrix representation of the tensors
  constexpr int64_t l_m = 3*8*6;
  constexpr int64_t l_n = 5*4;
  constexpr int64_t l_k = 2*7;

  constexpr int64_t l_sizes_a_mat = l_k*l_m;
  constexpr int64_t l_sizes_b_mat = l_n*l_k;
  constexpr int64_t l_sizes_c_mat = l_n*l_m;

  float l_a_mat[l_sizes_a_mat] = {0};
  float l_b_mat[l_sizes_b_mat] = {0};
  float l_c_mat[l_sizes_c_mat] = {0};

  // init values
  Catch::Generators::RandomFloatingGenerator< float > l_ran( -1.0f, 1.0f );

  for( int64_t l_en = 0; l_en < l_sizes_a_mat; l_en++ ) {
    l_a_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  for( int64_t l_en = 0; l_en < l_sizes_b_mat; l_en++ ) {
    l_b_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  for( int64_t l_en = 0; l_en < l_sizes_c_mat; l_en++ ) {
    l_c_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  // tensor representation
  float l_a_ten[l_sizes_a_mat] = {0};
  float l_b_ten[l_sizes_b_mat] = {0};
  float l_c_ten[l_sizes_c_mat] = {0};
  float l_c_ten_ref[l_sizes_c_mat] = {0};

  // strides A:
  //  matrix:            M - K
  //            3   8   6  -   2   7
  //    own:    1   3  24  - 144 288
  //
  //  tensor:   m0 m1  k0     k1  m2
  //            3   8   2      7   6
  //    own:    1   3  24     48 336
  //
  //    mat:    m0 m1  m2     k0  k1
  //            1   3 336     24  48
  int64_t l_sizes_tensor_in_left[5] = {7, 2, 6, 8, 3};
  int64_t l_strides_tensor_in_left_mat[5]  = {288, 144, 24, 3, 1};
  int64_t l_strides_tensor_in_left_native[5] = {48, 24, 336, 3, 1};

  transpose( 5,
             l_sizes_tensor_in_left,
             l_strides_tensor_in_left_mat,
             l_strides_tensor_in_left_native,
             l_a_mat,
             l_a_ten );

  // strides B:
  //  matrix:      K - N
  //            2  7 -  5  4
  //    own:    1  2 - 14 70
  //
  //  tensor:   k0 n0  k1 n1
  //             2  5   7  4
  //    own:     1  2  10 70
  //
  //    mat:    k0  k1 n0 n1
  //             1  10  2 70
  int64_t l_sizes_tensor_in_right[4] = {4, 5, 7, 2};
  int64_t l_strides_tensor_in_right_mat[4]  = {70, 14, 2, 1};
  int64_t l_strides_tensor_in_right_native[4] = {70, 2, 10, 1};

  transpose( 4,
             l_sizes_tensor_in_right,
             l_strides_tensor_in_right_mat,
             l_strides_tensor_in_right_native,
             l_b_mat,
             l_b_ten );

  // strides C:
  //  matrix:            M - N
  //            3   8   6  -   5   4
  //    own:    1   3  24  - 144 720
  //
  //  tensor:   m0  m1  n0  m2  n1
  //             3   8   5   6   4
  //    own:     1   3  24 120 720
  //
  //    mat:    m0  m1  m2  n0  n1
  //             1   3 120  24 720
  int64_t l_sizes_tensor_out[5] = {4, 5, 6, 8, 3};
  int64_t l_strides_tensor_out_mat[5]  = {720, 144, 24, 3, 1};
  int64_t l_strides_tensor_out_native[5] = {720, 24, 120, 3, 1};

  transpose( 5,
             l_sizes_tensor_out,
             l_strides_tensor_out_mat,
             l_strides_tensor_out_native,
             l_c_mat,
             l_c_ten );

  einsum_ir::backend::ContractionLoopsSimple l_cont_loops;
  l_cont_loops.init( l_num_dims_c,
                     l_num_dims_m,
                     l_num_dims_n,
                     l_num_dims_k,
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
                     l_strides_out_c,
                     l_strides_out_m,
                     l_strides_out_n,
                     4,
                     4,
                     4,
                     nullptr,
                     kernel_madd_fp32,
                     nullptr );

  l_cont_loops.contract( l_a_ten,
                         l_b_ten,
                         l_c_ten );

  gemm_ref_mnk( l_a_mat,
                l_b_mat,
                l_c_mat,
                l_m,
                l_n,
                l_k,
                l_m,
                l_k,
                l_m );

  transpose( 5,
             l_sizes_tensor_out,
             l_strides_tensor_out_mat,
             l_strides_tensor_out_native,
             l_c_mat,
             l_c_ten_ref );

  for( int64_t l_en = 0; l_en < l_sizes_c_mat; l_en++ ) {
    REQUIRE( l_c_ten_ref[l_en] == Approx( l_c_ten[l_en] ) );
  }
}

TEST_CASE( "Nested loops used in binary contractions using a matrix kernel.", "[contraction_loops_matrix]" ) {
  // Test case:
  //
  //   A: mb-kb-cb-bk-bm-bc
  //   B: kb-nb-cb-bk-bn-bc
  //
  //         ______________iexfghy________________
  //        /                                     \
  //   ie-a-x-c-g-y                            a-f-x-c-h-y
  //
  //   sizes:
  //     i: 3 (m0)
  //     e: 8 (m1)
  //     a: 2 (k0)
  //     c: 7 (k1)
  //     g: 6 (m2)
  //     f: 5 (n0)
  //     h: 4 (n1)
  //     x: 3 (c0)
  //     y: 4 (c1)
  //
  //     c: yx
  //     m: gei
  //     n: hf
  //     k: ca
  int64_t l_num_dims_c = 2;
  int64_t l_num_dims_m = 3;
  int64_t l_num_dims_n = 2;
  int64_t l_num_dims_k = 2;

  int64_t l_sizes_c[2] = { 4, 3 }; 
  int64_t l_sizes_m[3] = { 6, 8, 3 };
  int64_t l_sizes_n[2] = { 4, 5 };
  int64_t l_sizes_k[2] = { 7, 2 };

  // in left:
  //   sizes:       4 (y), 6 (g), 7 (c), 3 (x), 2 (a), 8 (e), 3 (i)
  //   strides:      6048,  1008,   144,    48,    24,    3,     1
  int64_t l_strides_in_left_c[2] = { 6048, 48 };
  int64_t l_strides_in_left_m[3] = { 1008, 3, 1 };
  int64_t l_strides_in_left_k[2] = { 144, 24 };

  // in right:
  //   sizes:       4 (y), 4 (h), 7 (c), 3 (x), 5 (f), 2 (a)
  //   strides:       840,   210,    30,    10,     2,     1
  int64_t l_strides_in_right_c[2] = { 840, 10 };
  int64_t l_strides_in_right_n[2] = { 210, 2 };
  int64_t l_strides_in_right_k[2] = { 30, 1 };

  // out:
  //   sizes:        4 (y), 4 (h), 6 (g), 5 (f), 3 (x), 8 (e), 3 (i)
  //   strides:       8640,  2160,   360,    72,    24,     3,     1
  int64_t l_strides_out_c[2] = { 8640, 24 };
  int64_t l_strides_out_m[3] = { 360, 3, 1 };
  int64_t l_strides_out_n[2] = { 2160, 72 };

  // column-major matrix representation of the tensors
  constexpr int64_t l_c = 3*4;
  constexpr int64_t l_m = 3*8*6;
  constexpr int64_t l_n = 5*4;
  constexpr int64_t l_k = 2*7;

  constexpr int64_t l_sizes_a_mat = l_c*l_k*l_m;
  constexpr int64_t l_sizes_b_mat = l_c*l_n*l_k;
  constexpr int64_t l_sizes_c_mat = l_c*l_n*l_m;

  float l_a_mat[l_sizes_a_mat] = {0};
  float l_b_mat[l_sizes_b_mat] = {0};
  float l_c_mat[l_sizes_c_mat] = {0};

  // init values
  Catch::Generators::RandomFloatingGenerator< float > l_ran( -1.0f, 1.0f );

  for( int64_t l_en = 0; l_en < l_sizes_a_mat; l_en++ ) {
    l_a_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  for( int64_t l_en = 0; l_en < l_sizes_b_mat; l_en++ ) {
    l_b_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  for( int64_t l_en = 0; l_en < l_sizes_c_mat; l_en++ ) {
    l_c_mat[l_en] = l_ran.get();
    l_ran.next();
  }

  // tensor representation
  float l_a_ten[l_sizes_a_mat] = {0};
  float l_b_ten[l_sizes_b_mat] = {0};
  float l_c_ten[l_sizes_c_mat] = {0};
  float l_c_ten_ref[l_sizes_c_mat] = {0};

  // strides A (ie-a-x-c-g-y):
  //   matrix
  //     4 (c1) 3 (c0) - 7 (k1) 2 (k0) - 6 (m2) 8 (m1) 3 (m0)
  //       6048   2016 -    288    144 -     24      3      1
  //
  //   tensor own
  //     4 (c1) 6 (m2)  7 (k1) 3 (c0)  2 (k0) 8 (m1) 3 (m0)
  //       6048   1008     144     48      24      3      1
  //   tensor mat
  //       6048 48 - 144 24 - 1008 3 1
  int64_t l_sizes_tensor_in_left[7] = {4, 3, 7, 2, 6, 8, 3};
  int64_t l_strides_tensor_in_left_mat[7]  = {6048, 2016, 288, 144, 24, 3, 1};
  int64_t l_strides_tensor_in_left_native[7] = { 6048, 48, 144, 24, 1008, 3, 1};

  transpose( 7,
             l_sizes_tensor_in_left,
             l_strides_tensor_in_left_mat,
             l_strides_tensor_in_left_native,
             l_a_mat,
             l_a_ten );

  // strides B (a-f-x-c-h-y)
  //   matrix
  //     4 (c1) 3 (c0) - 4 (n1) 5 (n0) - 7 (k1) 2 (k0)
  //        840    280       70     14        2      1
  //
  //   tensor own
  //     4 (c1) 4 (n1) 7 (k1) 3 (c0) 5 (n0) 2 (k0)
  //        840    210     30     10      2      1
  //   tensor mat
  //     840 10 - 210 2 - 30 1
  int64_t l_sizes_tensor_in_right[6] = {4, 3, 4, 5, 7, 2};
  int64_t l_strides_tensor_in_right_mat[6]  = {840, 280, 70, 14, 2, 1};
  int64_t l_strides_tensor_in_right_native[6] = {840, 10, 210, 2, 30, 1};

  transpose( 6,
             l_sizes_tensor_in_right,
             l_strides_tensor_in_right_mat,
             l_strides_tensor_in_right_native,
             l_b_mat,
             l_b_ten );


  // strides C (iexfghy)
  //   matrix
  //     4 (c1) 3 (c0) - 4 (n1) 5 (n0) - 6 (m2) 8 (m1) 3 (m0)
  //       8640   2880      720    144       24      3      1
  //
  //   tensor own
  //      4 (c1)  4 (n1) 6 (m2)  5 (n0) 3 (c0) 8 (m1) 3 (m0)
  //        8640    2160    360      72     24      3      1
  //   tensor mat
  //     8640 24 - 2160 72 - 360 3 1
  int64_t l_sizes_tensor_out[7] = {4, 3, 4, 5, 6, 8, 3};
  int64_t l_strides_tensor_out_mat[7]  = {8640, 2880, 720, 144, 24, 3, 1};
  int64_t l_strides_tensor_out_native[7] = {8640, 24, 2160, 72, 360, 3, 1};

  transpose( 7,
             l_sizes_tensor_out,
             l_strides_tensor_out_mat,
             l_strides_tensor_out_native,
             l_c_mat,
             l_c_ten );

  einsum_ir::backend::ContractionLoopsSimple l_cont_loops;
  l_cont_loops.init( l_num_dims_c,
                     l_num_dims_m-2,
                     l_num_dims_n-1,
                     l_num_dims_k-1,
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
                     l_strides_out_c,
                     l_strides_out_m,
                     l_strides_out_n,
                     4,
                     4,
                     4,
                     nullptr,
                     kernel_mat_24_5_2_24_2_72_fp32,
                     nullptr );

  l_cont_loops.contract( l_a_ten,
                         l_b_ten,
                         l_c_ten );

  for( int64_t l_ba = 0; l_ba < 4*3; l_ba++ ) {
    int64_t l_off_a = 2016 * l_ba;
    int64_t l_off_b =  280 * l_ba;
    int64_t l_off_c = 2880 * l_ba;
    gemm_ref_mnk( l_a_mat + l_off_a,
                  l_b_mat + l_off_b,
                  l_c_mat + l_off_c,
                  l_m,
                  l_n,
                  l_k,
                  l_m,
                  l_k,
                  l_m );
  }

  transpose( 7,
             l_sizes_tensor_out,
             l_strides_tensor_out_mat,
             l_strides_tensor_out_native,
             l_c_mat,
             l_c_ten_ref );

  for( int64_t l_en = 0; l_en < l_sizes_c_mat; l_en++ ) {
    REQUIRE( l_c_ten_ref[l_en] == Approx( l_c_ten[l_en] ) );
  }
}