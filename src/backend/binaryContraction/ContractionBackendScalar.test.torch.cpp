#include "ATen/ATen.h"
#include "catch.hpp"
#include "ContractionBackendScalar.h"


TEST_CASE( "Simple FP32 matmul using the Scalar contraction backend implementation.", "[contraction_backend_scalar]" ) {
  // Test Case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  using namespace einsum_ir;
  using namespace einsum_ir::backend;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 m, n, k mp,np,kp
  std::vector< int64_t > m_loop_sizes           = {  2, 3, 4, 1, 1, 1};  
  std::vector< int64_t > l_loop_strides_left    = {  1, 0, 2, 1, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = {  0, 4, 1, 0, 1, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0, 0, 0};
  std::vector< int64_t > l_loop_strides_out     = {  1, 2, 0, 1, 1, 0};

  einsum_ir::backend::ContractionBackendScalar l_bin_cont;
  l_bin_cont.init( l_loop_dim_type,
                   l_loop_exec_type,
                   m_loop_sizes,
                   l_loop_strides_left,
                   l_loop_strides_right,
                   l_loop_strides_out_aux,
                   l_loop_strides_out,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   kernel_t::UNDEFINED_KTYPE,
                   kernel_t::MADD,
                   kernel_t::UNDEFINED_KTYPE );   
  // data
  at::Tensor l_in_left  = at::rand( {4, 2} );
  at::Tensor l_in_right = at::rand( {3, 4} );
  at::Tensor l_out_ref  = at::rand( {3, 2} );
  at::Tensor l_out = l_out_ref.clone();


  // reference
  l_out_ref += at::einsum( "km,nk->nm",
                           {l_in_left, l_in_right} );

  // native input dimensions
  l_bin_cont.compile();
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       nullptr,
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "Matrix-matrix multiplication with a full-tensor bias using the Scalar contraction backend implementation.", "[contraction_backend_scalar]" ) {
  // Test Case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  using namespace einsum_ir;
  using namespace einsum_ir::backend;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 m, n, k mp,np,kp
  std::vector< int64_t > m_loop_sizes           = {  2, 3, 4, 1, 1, 1};  
  std::vector< int64_t > l_loop_strides_left    = {  1, 0, 2, 1, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = {  0, 4, 1, 0, 1, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  1, 2, 0, 1, 1, 0};
  std::vector< int64_t > l_loop_strides_out     = {  1, 2, 0, 1, 1, 0};

  einsum_ir::backend::ContractionBackendScalar l_bin_cont;
  l_bin_cont.init( l_loop_dim_type,
                   l_loop_exec_type,
                   m_loop_sizes,
                   l_loop_strides_left,
                   l_loop_strides_right,
                   l_loop_strides_out_aux,
                   l_loop_strides_out,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   kernel_t::COPY,
                   kernel_t::MADD,
                   kernel_t::UNDEFINED_KTYPE );  

  // data
  at::Tensor l_in_left  = at::rand( {4, 2} );
  at::Tensor l_in_right = at::rand( {3, 4} );
  at::Tensor l_bias     = at::rand( {3, 2} );
  at::Tensor l_out_ref  = at::rand( {3, 2} );
  at::Tensor l_out      = at::rand( {3, 2} );

  // reference
  l_out_ref = l_bias + at::einsum( "km,nk->nm",
                                   {l_in_left, l_in_right} );

  // native input dimensions
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}


TEST_CASE( "Matrix-matrix multiplication with a bias (scalar to matrix bcast) using the Scalar contraction backend implementation..", "[contraction_backend_scalar]" ) {
  // Test Case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  using namespace einsum_ir;
  using namespace einsum_ir::backend;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 m, n, k mp,np,kp
  std::vector< int64_t > m_loop_sizes           = {  2, 3, 4, 1, 1, 1};  
  std::vector< int64_t > l_loop_strides_left    = {  1, 0, 2, 1, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = {  0, 4, 1, 0, 1, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 1, 1, 0};
  std::vector< int64_t > l_loop_strides_out     = {  1, 2, 0, 1, 1, 0};

  einsum_ir::backend::ContractionBackendScalar l_bin_cont;
  l_bin_cont.init( l_loop_dim_type,
                   l_loop_exec_type,
                   m_loop_sizes,
                   l_loop_strides_left,
                   l_loop_strides_right,
                   l_loop_strides_out_aux,
                   l_loop_strides_out,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   kernel_t::COPY,
                   kernel_t::MADD,
                   kernel_t::UNDEFINED_KTYPE );

  // data
  at::Tensor l_in_left  = at::rand( {4, 2} );
  at::Tensor l_in_right = at::rand( {3, 4} );
  at::Tensor l_bias     = at::rand( {1, 1} );
  at::Tensor l_out_ref  = at::rand( {3, 2} );
  at::Tensor l_out      = at::rand( {3, 2} );

  // reference
  l_out_ref = l_bias + at::einsum( "km,nk->nm",
                                   {l_in_left, l_in_right} );

  // native input dimensions
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}


TEST_CASE( "Matrix-matrix multiplication with a bias (row to matrix bcast) using the Scalar contraction backend implementation.", "[contraction_backend_scalar]" ) {
  // Test Case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  using namespace einsum_ir;
  using namespace einsum_ir::backend;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 m, n, k mp,np,kp
  std::vector< int64_t > m_loop_sizes           = {  2, 3, 4, 1, 1, 1};  
  std::vector< int64_t > l_loop_strides_left    = {  1, 0, 2, 1, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = {  0, 4, 1, 0, 1, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 1, 0, 1, 1, 0};
  std::vector< int64_t > l_loop_strides_out     = {  1, 2, 0, 1, 1, 0};

  einsum_ir::backend::ContractionBackendScalar l_bin_cont;
  l_bin_cont.init( l_loop_dim_type,
                   l_loop_exec_type,
                   m_loop_sizes,
                   l_loop_strides_left,
                   l_loop_strides_right,
                   l_loop_strides_out_aux,
                   l_loop_strides_out,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   kernel_t::COPY,
                   kernel_t::MADD,
                   kernel_t::UNDEFINED_KTYPE );
  // data
  at::Tensor l_in_left  = at::rand( {4, 2} );
  at::Tensor l_in_right = at::rand( {3, 4} );
  at::Tensor l_bias     = at::rand( {3, 1} );
  at::Tensor l_out_ref  = at::rand( {3, 2} );
  at::Tensor l_out      = at::rand( {3, 2} );

  // reference
  l_out_ref = l_bias + at::einsum( "km,nk->nm",
                                   {l_in_left, l_in_right} );

  // native input dimensions
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}


TEST_CASE( "Matrix-matrix multiplication with a bias (column to matrix bcast) using the Scalar contraction backend implementation.", "[contraction_backend_scalar]" ) {
  // Test Case:
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  using namespace einsum_ir;
  using namespace einsum_ir::backend;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 m, n, k mp,np,kp
  std::vector< int64_t > m_loop_sizes           = {  2, 3, 4, 1, 1, 1};  
  std::vector< int64_t > l_loop_strides_left    = {  1, 0, 2, 1, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = {  0, 4, 1, 0, 1, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  1, 0, 0, 1, 1, 0};
  std::vector< int64_t > l_loop_strides_out     = {  1, 2, 0, 1, 1, 0};

  einsum_ir::backend::ContractionBackendScalar l_bin_cont;
  l_bin_cont.init( l_loop_dim_type,
                   l_loop_exec_type,
                   m_loop_sizes,
                   l_loop_strides_left,
                   l_loop_strides_right,
                   l_loop_strides_out_aux,
                   l_loop_strides_out,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   kernel_t::COPY,
                   kernel_t::MADD,
                   kernel_t::UNDEFINED_KTYPE );

  // data
  at::Tensor l_in_left  = at::rand( {4, 2} );
  at::Tensor l_in_right = at::rand( {3, 4} );
  at::Tensor l_bias     = at::rand( {1, 2} );
  at::Tensor l_out_ref  = at::rand( {3, 2} );
  at::Tensor l_out      = at::rand( {3, 2} );

  // reference
  l_out_ref = l_bias + at::einsum( "km,nk->nm",
                                   {l_in_left, l_in_right} );

  // native input dimensions
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_bias.data_ptr(),
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "Binary contraction involving C, M, N and K dimensions using the Scalar contraction backend implementation.", "[contraction_backend_scalar]" ) {
  // Test case:
  //
  //         ______________yhgfxei________________
  //        /                                     \
  //   ygcxaei                                   yhcxfa
  //
  //   char id size type
  //      i  0    3   m0
  //      e  1    8   m1
  //      a  2    2   k0
  //      c  3    7   k1
  //      g  4    6   m2
  //      f  5    5   n0
  //      h  6    4   n1
  //      x  7    3   c0
  //      y  8    4   c1
  //
  //  ieaxcgy: 8 4 3 7 2 1 0
  //  afxchy:  8 6 3 7 5 2
  //  iexfghy: 8 6 4 5 7 1 0
  //
  //   dim types:
  //     c:  yx /  87
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32

  using namespace einsum_ir;
  using namespace einsum_ir::backend;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::M, 
                                             dim_t::K,
                                             dim_t::K, 
                                             dim_t::M, 
                                             dim_t::N,
                                             dim_t::N, 
                                             dim_t::C, 
                                             dim_t::C,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 i, e, a,  c,   g, f,   h, x,   y,mp,np,kp
  std::vector< int64_t > m_loop_sizes           = {  3, 8, 2,  7,   6, 5,   4, 3,   4, 1, 1, 1};  
  std::vector< int64_t > l_loop_strides_left    = {  1, 3,24,144,1008, 0,   0,48,6048, 1, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = {  0, 0, 1, 30,   0, 2, 210,10, 840, 0, 1, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0,  0,   0, 0,   0, 0,   0, 0, 0, 0};
  std::vector< int64_t > l_loop_strides_out     = {  1, 3, 0,  0, 360,72,2160,24,8640, 1, 1, 0};

  einsum_ir::backend::ContractionBackendScalar l_bin_cont;
  l_bin_cont.init( l_loop_dim_type,
                   l_loop_exec_type,
                   m_loop_sizes,
                   l_loop_strides_left,
                   l_loop_strides_right,
                   l_loop_strides_out_aux,
                   l_loop_strides_out,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   data_t::FP32,
                   kernel_t::UNDEFINED_KTYPE,
                   kernel_t::MADD,
                   kernel_t::UNDEFINED_KTYPE );

  //                                0  1  2  3  4  5  6
  //                                y  g  c  x  a  e  i
  at::Tensor l_in_left = at::rand( {4, 6, 7, 3, 2, 8, 3} );
  //                                 0  1  2  3  4  5
  //                                 y  h  c  x  f  a
  at::Tensor l_in_right = at::rand( {4, 4, 7, 3, 5, 2} );
  //                                y  h  g  f  x  e  i
  at::Tensor l_out_ref = at::rand( {4, 4, 6, 5, 3, 8, 3} );
  at::Tensor l_out = l_out_ref.clone();


  // reference
  l_out_ref += at::einsum( "ygcxaei,yhcxfa->yhgfxei",
                           {l_in_left, l_in_right} );

  l_bin_cont.compile();
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       nullptr,
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}


TEST_CASE( "Binary contraction involving C, M, N and K dimensions using FP64, zero and relu using the Scalar contraction backend implementation.", "[contraction_backend_scalar]" ) {
  // Test case:
  //
  //         ______________yhgfxei________________
  //        /                                     \
  //   ygcxaei                                   yhcxfa
  //
  //   char id size type
  //      i  0    3   m0
  //      e  1    8   m1
  //      a  2    2   k0
  //      c  3    7   k1
  //      g  4    6   m2
  //      f  5    5   n0
  //      h  6    4   n1
  //      x  7    3   c0
  //      y  8    4   c1
  //
  //  ieaxcgy: 8 4 3 7 2 1 0
  //  afxchy:  8 6 3 7 5 2
  //  iexfghy: 8 6 4 5 7 1 0
  //
  //   dim types:
  //     c:  yx /  87
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32

  using namespace einsum_ir;
  using namespace einsum_ir::backend;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M, 
                                             dim_t::M, 
                                             dim_t::K,
                                             dim_t::K, 
                                             dim_t::M, 
                                             dim_t::N,
                                             dim_t::N, 
                                             dim_t::C, 
                                             dim_t::C,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::SEQ, 
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 i, e, a,  c,   g, f,   h, x,   y,mp,np,kp
  std::vector< int64_t > m_loop_sizes           = {  3, 8, 2,  7,   6, 5,   4, 3,   4, 1, 1, 1};  
  std::vector< int64_t > l_loop_strides_left    = {  1, 3,24,144,1008, 0,   0,48,6048, 1, 0, 1};
  std::vector< int64_t > l_loop_strides_right   = {  0, 0, 1, 30,   0, 2, 210,10, 840, 0, 1, 1};
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0,  0,   0, 0,   0, 0,   0, 0, 0, 0};
  std::vector< int64_t > l_loop_strides_out     = {  1, 3, 0,  0, 360,72,2160,24,8640, 1, 1, 0};

  einsum_ir::backend::ContractionBackendScalar l_bin_cont;
  l_bin_cont.init( l_loop_dim_type,
                   l_loop_exec_type,
                   m_loop_sizes,
                   l_loop_strides_left,
                   l_loop_strides_right,
                   l_loop_strides_out_aux,
                   l_loop_strides_out,
                   data_t::FP64,
                   data_t::FP64,
                   data_t::FP64,
                   data_t::FP64,
                   kernel_t::ZERO,
                   kernel_t::MADD,
                   kernel_t::RELU );

  //                                0  1  2  3  4  5  6
  //                                y  g  c  x  a  e  i
  at::Tensor l_in_left = at::rand( {4, 6, 7, 3, 2, 8, 3},
                                   at::ScalarType::Double );
  //                                 0  1  2  3  4  5
  //                                 y  h  c  x  f  a
  at::Tensor l_in_right = at::rand( {4, 4, 7, 3, 5, 2},
                                    at::ScalarType::Double );
  //                                y  h  g  f  x  e  i
  at::Tensor l_out_ref = at::rand( {4, 4, 6, 5, 3, 8, 3},
                                   at::ScalarType::Double );
  at::Tensor l_out = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "ygcxaei,yhcxfa->yhgfxei",
                          {l_in_left, l_in_right} );

  l_out_ref = at::relu( l_out_ref );

  l_bin_cont.compile();
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       nullptr,
                       l_out.data_ptr() );

  REQUIRE( at::allclose( l_out, l_out_ref )  );
}
