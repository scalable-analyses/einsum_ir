#include "ATen/ATen.h"
#include "catch.hpp"
#include "ContractionBackendTpp.h"

TEST_CASE( "Matmul with sequential batch dimension.", "[contraction_backend]" ) {
  //example: [c1,k1,m1],[c1,n1,k1]->[c1,n1,m1]
  //sizes:   [17,13,20],[17,47,13]->[17,47,20]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 c1,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {  17,20,47,13 };  
  std::vector< int64_t > l_loop_strides_left    = { 260, 1, 0,20 };
  std::vector< int64_t > l_loop_strides_right   = { 611, 0,13, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {   0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = { 940, 1,20, 0 };

  at::Tensor l_left    = at::randn( { 17,13,20 } );
  at::Tensor l_right   = at::randn( { 17,47,13 } );
  at::Tensor l_out     = at::zeros( { 17,47,20 } );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::ZERO,
               kernel_t::MADD,
               kernel_t::UNDEFINED_KTYPE,
               2 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );


  l_out_ref = at::einsum( "xcb,xac->xab",
                          { l_left, l_right } );
  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Packed Matmul with sequential M dimension.", "[contraction_backend]" ) {
  //example: [m2,k1,m1,c1],[n1,k1,c1]->[m2,n1,m1,c1]
  //sizes:   [ 5,13,20,17],[47,13,17]->[ 5,47,20,17]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M,
                                             dim_t::C,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ,
                                             exec_t::PRIM,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                   m2, c1, m1, n1, k1
  std::vector< int64_t > m_loop_sizes           = {     5, 17, 20, 47, 13 };  
  std::vector< int64_t > l_loop_strides_left    = {  4420,  1, 17,  0,340 };
  std::vector< int64_t > l_loop_strides_right   = {     0,  1,  0,221, 17 };
  std::vector< int64_t > l_loop_strides_out_aux = {     0,  0,  0,  0,  0 };
  std::vector< int64_t > l_loop_strides_out     = { 15980,  1, 17,340,  0 };

  at::Tensor l_left    = at::randn( { 5,13,20,17 } );
  at::Tensor l_right   = at::randn( {   47,13,17 } );
  at::Tensor l_out     = at::ones( { 5,47,20,17 } );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::ZERO,
               kernel_t::PACKED_MADD,
               kernel_t::UNDEFINED_KTYPE,
               3 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );

  l_out_ref = at::einsum( "dcbx,acx->dabx",
                          { l_left, l_right } );
  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Matmul with sequential batch dimension and transposed B.", "[contraction_backend]" ) {
  //example: [c1,k1,m1],[c1,k1,n1]->[c1,n1,m1]
  //sizes:   [ 5, 3, 2],[ 5, 3, 4]->[ 5, 4, 2]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                c1,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {  5, 2, 4, 3 };  
  std::vector< int64_t > l_loop_strides_left    = {  6, 1, 0, 2 };
  std::vector< int64_t > l_loop_strides_right   = { 12, 0, 1, 4 };
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = {  8, 1, 2, 0 };

  at::Tensor l_left    = at::randn( { 5,3,2 } );
  at::Tensor l_right   = at::randn( { 5,3,4 } );
  at::Tensor l_out     = at::zeros( { 5,4,2 } );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::ZERO,
               kernel_t::MADD,
               kernel_t::UNDEFINED_KTYPE,
               4 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );

  l_out_ref = at::einsum( "xcb,xca->xab",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Simple Matmul with sequential batch dimension and transposed A.", "[contraction_backend]" ) {
  //example: [c1,m1,k1],[c1,n1,k1]->[c1,n1,m1]
  //sizes:   [ 5, 2, 3],[ 5, 4, 3]->[ 5, 4, 2]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                c1,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {  5, 2, 4, 3 };  
  std::vector< int64_t > l_loop_strides_left    = {  6, 3, 0, 1 };
  std::vector< int64_t > l_loop_strides_right   = { 12, 0, 3, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = {  8, 1, 2, 0 };

  at::Tensor l_left    = at::randn( { 5,2,3 } );
  at::Tensor l_right   = at::randn( { 5,4,3 } );
  at::Tensor l_out     = at::zeros( { 5,4,2 } );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::ZERO,
               kernel_t::MADD,
               kernel_t::UNDEFINED_KTYPE,
               5 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );

  l_out_ref = at::einsum( "xbc,xac->xab",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}


TEST_CASE( "Single call of batch reduce matmul.", "[contraction_backend]" ) {
  //example: [k2,k1,m1],[k2,n1,k1]->[n1,m1]
  //sizes:   [ 2, 7, 5],[ 2, 4, 7]->[ 4, 5]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::K,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::PRIM,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                k2,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {  3, 5, 4, 7 };  
  std::vector< int64_t > l_loop_strides_left    = { 35, 1, 0, 5 };
  std::vector< int64_t > l_loop_strides_right   = { 28, 0, 7, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = {  0, 1, 5, 0 };

  at::Tensor l_left    = at::randn( {  3, 7, 5 } );
  at::Tensor l_right   = at::randn( {  3, 4, 7 } );
  at::Tensor l_out     = at::zeros( {     4, 5 } );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::ZERO,
               kernel_t::BR_MADD,
               kernel_t::UNDEFINED_KTYPE,
               6 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );

  l_out_ref = at::einsum( "xcb,xac->ab",
                          { l_left, l_right } );
  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Tensor contraction with omp parallelisation.", "[contraction_backend]" ) {
  //example: [m1,k1,m1],[n2,n1,k1]->[n2,m1,n1,m1]
  //sizes:   [17,13,20],[ 8,47,13]->[ 8,17,47,20]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::N,
                                             dim_t::M,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::OMP,
                                             exec_t::OMP,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                   n2, m2,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {     8, 17,20,47,13 };  
  std::vector< int64_t > l_loop_strides_left    = {     0,260, 1, 0,20 };
  std::vector< int64_t > l_loop_strides_right   = {   611,  0, 0,13, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {     0,  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = { 15980,940, 1,20, 0 };

  at::Tensor l_left    = at::randn( {   17,13,20 } );
  at::Tensor l_right   = at::randn( {    8,47,13 } );
  at::Tensor l_out     = at::zeros( { 8,17,47,20 } );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::ZERO,
               kernel_t::MADD,
               kernel_t::UNDEFINED_KTYPE,
               7 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );


  l_out_ref = at::einsum( "xcb,yac->yxab",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Blocked matmul with omp parallelisation.", "[contraction_backend]" ) {
  //example: [m2,k2,k1,m1],[n2,k2,n1,k1]->[n2,m2,n1,m1]
  //sizes:   [32, 8,64,64],[32, 8,64,64]->[32,32,64,64]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::N,
                                             dim_t::M,
                                             dim_t::K, 
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::OMP,
                                             exec_t::OMP,
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                    n2,   m2,  k2,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {     32,   32,   8,64,64,64 };  
  std::vector< int64_t > l_loop_strides_left    = {      0,32768,4096, 1, 0,64 };
  std::vector< int64_t > l_loop_strides_right   = {  32768,    0,4096, 0,64, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {      0,    0,   0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = { 131072, 4096,   0, 1,64, 0 };

  at::Tensor l_left    = at::randn( { 32, 8,64,64 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_right   = at::randn( { 32, 8,64,64 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out     = at::randn( { 32,32,64,64 },
                                    at::dtype( at::kDouble ) );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::UNDEFINED_KTYPE,
               8 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );

  l_out_ref = at::einsum( "abcd,ebfc->eafd",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref ) );
}

TEST_CASE( "Tensor contraction with SFC parallelisation.", "[contraction_backend]" ) {
  //example: [m1,k1,m1],[n2,n1,k1]->[n2,m1,n1,m1]
  //sizes:   [17,13,20],[ 8,47,13]->[ 8,17,47,20]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::M,
                                             dim_t::N,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::SFC,
                                             exec_t::SFC,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                 m2,   n2,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {  17,    8,20,47,13 };  
  std::vector< int64_t > l_loop_strides_left    = { 260,    0, 1, 0,20 };
  std::vector< int64_t > l_loop_strides_right   = {   0,  611, 0,13, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {   0,    0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = { 940,15980, 1,20, 0 };

  at::Tensor l_left    = at::randn( {   17,13,20 } );
  at::Tensor l_right   = at::randn( {    8,47,13 } );
  at::Tensor l_out     = at::zeros( { 8,17,47,20 } );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::ZERO,
               kernel_t::MADD,
               kernel_t::UNDEFINED_KTYPE,
               9 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );


  l_out_ref = at::einsum( "xcb,yac->yxab",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Tensor contraction with SFC and omp parallelisation.", "[contraction_backend]" ) {
  //example: [c1,m1,k1,m1],[c1,n2,n1,k1]->[c1,n2,m1,n1,m1]
  //sizes:   [ 5,17,13,20],[ 5, 8,47,13]->[ 5, 8,17,47,20]
  using namespace etops;

  std::vector< dim_t >  l_loop_dim_type  = { dim_t::C,
                                             dim_t::M,
                                             dim_t::N,
                                             dim_t::M, 
                                             dim_t::N, 
                                             dim_t::K };
  std::vector< exec_t > l_loop_exec_type = { exec_t::OMP,
                                             exec_t::SFC,
                                             exec_t::SFC,
                                             exec_t::PRIM, 
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //                                                    c1,  m2,   n2,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {      5, 17,    8,20,47,13 };  
  std::vector< int64_t > l_loop_strides_left    = {   4420,260,    0, 1, 0,20 };
  std::vector< int64_t > l_loop_strides_right   = {   4888,  0,  611, 0,13, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {      0,  0,    0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = { 127840,940,15980, 1,20, 0 };

  at::Tensor l_left    = at::randn( {   5,17,13,20 } );
  at::Tensor l_right   = at::randn( {   5, 8,47,13 } );
  at::Tensor l_out     = at::zeros( { 5,8,17,47,20 } );
  at::Tensor l_out_ref = l_out.clone();

  binary::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
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
               kernel_t::ZERO,
               kernel_t::MADD,
               kernel_t::UNDEFINED_KTYPE,
               10 );     
                
  err_t l_err = l_cont.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );


  l_out_ref = at::einsum( "zxcb,zyac->zyxab",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}