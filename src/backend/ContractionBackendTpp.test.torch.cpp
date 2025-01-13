#include "ATen/ATen.h"
#include "catch.hpp"
#include "ContractionBackendTpp.h"
#include "../constants.h"

#include <iostream>

TEST_CASE( "Simple Bached Matmul with sequential batch dimension.", "[contraction_backend]" ) {
  //example: [c1,k1,m1],[c1,n1,k1]->[c1,n1,m1]
  //sizes:   [17,13,20],[17,47,13]->[17,47,20]

  std::vector< einsum_ir::dim_t >  l_loop_dim_type  = { einsum_ir::dim_t::C,
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::K };
  std::vector< einsum_ir::exec_t > l_loop_exec_type = { einsum_ir::exec_t::SEQ,
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM };

  //                                                c1,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = { 17,20,47,13 };  
  std::vector< int64_t > l_loop_strides_left    = {260, 1, 0,20 };
  std::vector< int64_t > l_loop_strides_right   = {611, 0,13, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {  0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = {940, 1,20, 0 };

  at::Tensor l_left    = at::randn( { 17,13,20 } );
  at::Tensor l_right   = at::randn( { 17,47,13 } );
  at::Tensor l_out     = at::zeros( { 17,47,20 } );
  at::Tensor l_out_ref = l_out.clone();

  einsum_ir::backend::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
               l_loop_exec_type,
               m_loop_sizes,
               l_loop_strides_left,
               l_loop_strides_right,
               l_loop_strides_out_aux,
               l_loop_strides_out,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::kernel_t::ZERO,
               einsum_ir::kernel_t::MADD,
               einsum_ir::kernel_t::UNDEFINED_KTYPE );     
                
  einsum_ir::err_t l_err = l_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );


  l_out_ref = at::einsum( "xcb,xac->xab",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "Single call of batch reduce matmul.", "[contraction_backend2]" ) {
  //example: [k2,k1,m1],[k2,n1,k1]->[n1,m1]
  //sizes:   [ 2, 7, 5],[ 2, 4, 7]->[ 4, 5]

  std::vector< einsum_ir::dim_t >  l_loop_dim_type  = { einsum_ir::dim_t::UNDEFINED_DIM,
                                                        einsum_ir::dim_t::K,
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::K };
  std::vector< einsum_ir::exec_t > l_loop_exec_type = { einsum_ir::exec_t::SEQ,
                                                        einsum_ir::exec_t::PRIM,
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM };

  //                                                k2,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = { 1, 3, 5, 4, 7 };  
  std::vector< int64_t > l_loop_strides_left    = { 0,35, 1, 0, 5 };
  std::vector< int64_t > l_loop_strides_right   = { 0,28, 0, 7, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = { 0, 0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = { 0, 0, 1, 5, 0 };

  at::Tensor l_left    = at::ones( {  3, 7, 5 } );
  at::Tensor l_right   = at::ones( {  3, 4, 7 } );
  at::Tensor l_out     = at::zeros( {     4, 5 } );
  at::Tensor l_out_ref = l_out.clone();

  einsum_ir::backend::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
               l_loop_exec_type,
               m_loop_sizes,
               l_loop_strides_left,
               l_loop_strides_right,
               l_loop_strides_out_aux,
               l_loop_strides_out,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::kernel_t::ZERO,
               einsum_ir::kernel_t::BR_MADD,
               einsum_ir::kernel_t::UNDEFINED_KTYPE );     
                
  einsum_ir::err_t l_err = l_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );

  l_out_ref = at::einsum( "xcb,xac->ab",
                          { l_left, l_right } );
  std::cout << l_out << std::endl;
  std::cout << l_out_ref << std::endl;
  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "tensor contraction with omp parallelisation.", "[contraction_backend]" ) {
  //example: [m1,k1,m1],[n2,n1,k1]->[n2,m1,n1,m1]
  //sizes:   [17,13,20],[ 8,47,13]->[ 8,17,47,20]

  std::vector< einsum_ir::dim_t >  l_loop_dim_type  = { einsum_ir::dim_t::N,
                                                        einsum_ir::dim_t::M,
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::K };
  std::vector< einsum_ir::exec_t > l_loop_exec_type = { einsum_ir::exec_t::OMP,
                                                        einsum_ir::exec_t::OMP,
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM };

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

  einsum_ir::backend::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
               l_loop_exec_type,
               m_loop_sizes,
               l_loop_strides_left,
               l_loop_strides_right,
               l_loop_strides_out_aux,
               l_loop_strides_out,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::kernel_t::ZERO,
               einsum_ir::kernel_t::MADD,
               einsum_ir::kernel_t::UNDEFINED_KTYPE );     
                
  einsum_ir::err_t l_err = l_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );


  l_out_ref = at::einsum( "xcb,yac->yxab",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "blocked matmul with omp parallelisation.", "[contraction_backend1]" ) {
  //example: [m2,k2,k1,m1],[n2,k2,n1,k1]->[n2,m2,n1,m1]
  //sizes:   [32, 8,64,64],[32, 8,64,64]->[32,32,64,64]

  std::vector< einsum_ir::dim_t >  l_loop_dim_type  = { einsum_ir::dim_t::N,
                                                        einsum_ir::dim_t::M,
                                                        einsum_ir::dim_t::K, 
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::K };
  std::vector< einsum_ir::exec_t > l_loop_exec_type = { einsum_ir::exec_t::OMP,
                                                        einsum_ir::exec_t::OMP,
                                                        einsum_ir::exec_t::SEQ,
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM };

  //                                                    n2,   m2,  k2,m1,n1,k1
  std::vector< int64_t > m_loop_sizes           = {     32,   32,   8,64,64,64 };  
  std::vector< int64_t > l_loop_strides_left    = {      0,32768,4096, 1, 0,64 };
  std::vector< int64_t > l_loop_strides_right   = {  32768,    0,4096, 0,64, 1 };
  std::vector< int64_t > l_loop_strides_out_aux = {      0,    0,   0, 0, 0, 0 };
  std::vector< int64_t > l_loop_strides_out     = { 131072, 4096,   0, 1,64, 0 };

  at::Tensor l_left    = at::randn( { 32, 8,64,64 } );
  at::Tensor l_right   = at::randn( { 32, 8,64,64 } );
  at::Tensor l_out     = at::randn( { 32,32,64,64 } );
  at::Tensor l_out_ref = l_out.clone();

  einsum_ir::backend::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
               l_loop_exec_type,
               m_loop_sizes,
               l_loop_strides_left,
               l_loop_strides_right,
               l_loop_strides_out_aux,
               l_loop_strides_out,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::kernel_t::ZERO,
               einsum_ir::kernel_t::MADD,
               einsum_ir::kernel_t::UNDEFINED_KTYPE );     
                
  einsum_ir::err_t l_err = l_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  double l_time = 0;

  l_tp0 = std::chrono::steady_clock::now();
  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();

  std::cout << 1.0E-9 * 4290772992 / l_time << " " << l_time << std::endl;

  //example: [m2,k2,k1,m1],[n2,k2,n1,k1]->[n2,m2,n1,m1]
  l_out_ref = at::einsum( "abcd,ebfc->eafd",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}

TEST_CASE( "tensor contraction with sfc parallelisation and a jump in sfc.", "[contraction_backend]" ) {
  //example: [m1,k1,m1],[n2,n1,k1]->[n2,m1,n1,m1]
  //sizes:   [17,13,20],[ 8,47,13]->[ 8,17,47,20]

  std::vector< einsum_ir::dim_t >  l_loop_dim_type  = { einsum_ir::dim_t::N,
                                                        einsum_ir::dim_t::M,
                                                        einsum_ir::dim_t::M, 
                                                        einsum_ir::dim_t::N, 
                                                        einsum_ir::dim_t::K };
  std::vector< einsum_ir::exec_t > l_loop_exec_type = { einsum_ir::exec_t::SFC,
                                                        einsum_ir::exec_t::SFC,
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM, 
                                                        einsum_ir::exec_t::PRIM };

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

  einsum_ir::backend::ContractionBackendTpp l_cont;

  l_cont.init( l_loop_dim_type,
               l_loop_exec_type,
               m_loop_sizes,
               l_loop_strides_left,
               l_loop_strides_right,
               l_loop_strides_out_aux,
               l_loop_strides_out,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::data_t::FP32,
               einsum_ir::kernel_t::ZERO,
               einsum_ir::kernel_t::MADD,
               einsum_ir::kernel_t::UNDEFINED_KTYPE );     
                
  einsum_ir::err_t l_err = l_cont.compile();
  REQUIRE( l_err == einsum_ir::err_t::SUCCESS );

  l_cont.contract( l_left.data_ptr(),
                   l_right.data_ptr(),
                   nullptr,
                   l_out.data_ptr() );


  l_out_ref = at::einsum( "xcb,yac->yxab",
                          { l_left, l_right } );

  REQUIRE( at::allclose( l_out, l_out_ref, 1E-4, 1E-5 ) );
}
