#include "ATen/ATen.h"
#include "catch.hpp"
#include "UnaryBackendScalar.h"

TEST_CASE( "Scalar vector copy through the unary backend using FP64 data.", "[unary_backend_scalar]" ) {
  using namespace einsum_ir::etops;


  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  std::vector< int64_t > l_loop_sizes       = { 3,1,1 };  
  std::vector< int64_t > l_loop_strides_in  = { 1,1,1 };
  std::vector< int64_t > l_loop_strides_out = { 1,1,1 };


  UnaryBackendScalar l_unary_scalar;

  l_unary_scalar.init( l_loop_exec_type,
                       l_loop_sizes,
                       l_loop_strides_in,
                       l_loop_strides_out,
                       data_t::FP64,
                       data_t::FP64,
                       data_t::FP64,
                       kernel_t::COPY,
                       1 );     

  err_t l_err = l_unary_scalar.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  at::Tensor l_t0 = at::randn( {3},
                               at::ScalarType::Double );

  at::Tensor l_t1 = at::zeros( {3},
                               at::ScalarType::Double );

  l_unary_scalar.contract( l_t0.data_ptr(),
                        l_t1.data_ptr() );

  REQUIRE( at::equal( l_t0 , l_t1 ) );
}

TEST_CASE( "Scalar small tensor transposition through the unary backend using FP64 data.", "[unary_backend_scalar]" ) {
  using namespace einsum_ir::etops;

  std::vector< exec_t > l_loop_exec_type = { exec_t::SEQ,
                                             exec_t::SEQ, 
                                             exec_t::SEQ,
                                             exec_t::PRIM,
                                             exec_t::PRIM };

  std::vector< int64_t > l_loop_sizes       = { 5, 3, 4,1,1 };  
  std::vector< int64_t > l_loop_strides_in  = { 4,20, 1,1,1 };
  std::vector< int64_t > l_loop_strides_out = { 3, 1,15,1,1 };


  UnaryBackendScalar l_unary_scalar;

  l_unary_scalar.init( l_loop_exec_type,
                       l_loop_sizes,
                       l_loop_strides_in,
                       l_loop_strides_out,
                       data_t::FP64,
                       data_t::FP64,
                       data_t::FP64,
                       kernel_t::COPY,
                       1 );  

  err_t l_err = l_unary_scalar.compile();
  REQUIRE( l_err == err_t::SUCCESS );

  at::Tensor l_t0 = at::randn( {3, 5, 4},
                               at::ScalarType::Double );

  at::Tensor l_t1 = at::zeros( {4, 5, 3},
                               at::ScalarType::Double );

  l_unary_scalar.contract( l_t0.data_ptr(),
                        l_t1.data_ptr() );

  REQUIRE( at::equal( l_t0.permute( {2, 1, 0} ), l_t1 ) );
}

TEST_CASE( "Scalar large tensor transposition through the unary backend using FP32 data.", "[unary_backend_scalar]" ) {
  // dims_in   0, 1, 2, 3, 4, 5, 6, 7, 8 
  // dims_out  2, 1, 4, 0, 5, 7, 3, 8, 6 
  // sizes     0=3, 1=5, 2=4, 3=7, 4=2, 5=5, 6=3, 7=8, 8=6

  using namespace einsum_ir::etops;

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
                                             exec_t::PRIM };

  //dims                                             0,     1,      2,    3,     4,    5,   7,   6, 8
  std::vector< int64_t > l_loop_sizes       = {      3,     5,      4,    7,     2,    5,   8,   3, 6,1,1 };  
  std::vector< int64_t > l_loop_strides_in  = { 201600, 40320,  10080, 1440,   720,  144,   6,  48, 1,1,1 };
  std::vector< int64_t > l_loop_strides_out = {   5040, 30240, 151200,   18, 15120, 1008, 126,   1, 3,1,1 };


  UnaryBackendScalar l_unary_scalar;

  l_unary_scalar.init( l_loop_exec_type,
                    l_loop_sizes,
                    l_loop_strides_in,
                    l_loop_strides_out,
                    data_t::FP32,
                    data_t::FP32,
                    data_t::FP32,
                    kernel_t::COPY,
                    1 );  

  l_unary_scalar.compile();

  //                            0  1  2  3  4  5  6  7  8
  at::Tensor l_t0 = at::randn( {3, 5, 4, 7, 2, 5, 3, 8, 6},
                               at::ScalarType::Float );

  at::Tensor l_t1 = at::randn( {4, 5, 2, 3, 5, 8, 7, 6, 3},
                               at::ScalarType::Float );

  l_unary_scalar.contract( l_t0.data_ptr(),
                        l_t1.data_ptr() );

  REQUIRE( at::equal( l_t0.permute( {2, 1, 4, 0, 5, 7, 3, 8, 6} ), l_t1 ) );
}

TEST_CASE( "Scalar large tensor transposition through the unary backend with parallelization using FP32 data.", "[unary_backend_scalar]" ) {
  // dims_in   0, 1, 2, 3, 4, 5, 6, 7, 8 
  // dims_out  2, 1, 4, 0, 5, 7, 3, 8, 6 
  // sizes     0=3, 1=5, 2=4, 3=7, 4=2, 5=5, 6=3, 7=8, 8=6

  using namespace einsum_ir::etops;

  std::vector< exec_t > l_loop_exec_type = { exec_t::OMP,
                                             exec_t::OMP,
                                             exec_t::OMP,
                                             exec_t::OMP,
                                             exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::SEQ,
                                             exec_t::PRIM, 
                                             exec_t::PRIM };

  //dims                                             0,     1,      2,    3,     4,    5,   7,   6, 8
  std::vector< int64_t > l_loop_sizes       = {      3,     5,      4,    7,     2,    5,   8,   3, 6,1,1 };  
  std::vector< int64_t > l_loop_strides_in  = { 201600, 40320,  10080, 1440,   720,  144,   6,  48, 1,1,1 };
  std::vector< int64_t > l_loop_strides_out = {   5040, 30240, 151200,   18, 15120, 1008, 126,   1, 3,1,1 };


  UnaryBackendScalar l_unary_scalar;

  l_unary_scalar.init( l_loop_exec_type,
                    l_loop_sizes,
                    l_loop_strides_in,
                    l_loop_strides_out,
                    data_t::FP32,
                    data_t::FP32,
                    data_t::FP32,
                    kernel_t::COPY,
                    32 );  

  l_unary_scalar.compile();

  //                            0  1  2  3  4  5  6  7  8
  at::Tensor l_t0 = at::randn( {3, 5, 4, 7, 2, 5, 3, 8, 6},
                               at::ScalarType::Float );

  at::Tensor l_t1 = at::randn( {4, 5, 2, 3, 5, 8, 7, 6, 3},
                               at::ScalarType::Float );

  l_unary_scalar.contract( l_t0.data_ptr(),
                        l_t1.data_ptr() );

  REQUIRE( at::equal( l_t0.permute( {2, 1, 4, 0, 5, 7, 3, 8, 6} ), l_t1 ) );
}
