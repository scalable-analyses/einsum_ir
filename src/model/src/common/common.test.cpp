#include "catch.hpp"
#include "common.h"
#include "interpolation.h"

TEST_CASE( "Get time Model function generic.", "[common]" ) {
  using namespace einsum_ir::model::common;

  double gflops = 0.0;
  double time = get_time_model(128, 128, 128, 0, 0,einsum_ir::model::common::DType::FP32, Model::GENERIC, gflops, 1000.0, 16);
  REQUIRE( time > 0.0 );
  REQUIRE( gflops > 0.0 );
}

TEST_CASE( "Get time Model function zen5.", "[common]" ) {
  using namespace einsum_ir::model::common;

  double gflops = 0.0;
  double time = get_time_model(128, 128, 128, 0, 0,einsum_ir::model::common::DType::FP32, Model::ZEN5, gflops);
  REQUIRE( time > 0.0 );
  REQUIRE( gflops > 0.0 );
}

TEST_CASE( "Get time Model function m4.", "[common]" ) {
  using namespace einsum_ir::model::common;

  double gflops = 0.0;
  double time = get_time_model(128, 128, 128, 0, 0,einsum_ir::model::common::DType::FP32, Model::M4, gflops);
  REQUIRE( time > 0.0 );
  REQUIRE( gflops > 0.0 );
}

TEST_CASE( "Get time Model function a76.", "[common]" ) {
  using namespace einsum_ir::model::common;

  double gflops = 0.0;
  double time = get_time_model(128, 128, 128, 0, 0,einsum_ir::model::common::DType::FP32, Model::A76, gflops);
  REQUIRE( time > 0.0 );
  REQUIRE( gflops > 0.0 );
}

TEST_CASE( "Get time Model function bad input." "[common]") {
    using namespace einsum_ir::model::common;

    double gflops = 0.0;
    double time = get_time_model(-128, 128, 128, 0, 0,einsum_ir::model::common::DType::FP32, Model::GENERIC, gflops, 1000.0, 16);
    REQUIRE( time == 0.0 ); 
    REQUIRE( gflops == 0.0 );
}

TEST_CASE( "Linear interpolation between two values.", "[common]" ) {
  using namespace einsum_ir::model::common;

  double result = lerp(0.0, 10.0, 0.5);
  REQUIRE( result == 5.0 );
}

TEST_CASE( "Find bounds with interpolation in array.", "[common]" ) {
  using namespace einsum_ir::model::common;

  int arr[] = {10, 20, 30, 40, 50};
  int idx_lower = 0;
  double t = 0.0;

  find_bounds_with_interpolation(arr, sizeof(arr)/sizeof(arr[0]), 25, idx_lower, t);
  REQUIRE( idx_lower == 1 );
  REQUIRE( t == Approx(0.5) );

  find_bounds_with_interpolation(arr, sizeof(arr)/sizeof(arr[0]), 10, idx_lower, t);
  REQUIRE( idx_lower == 0 );
  REQUIRE( t == Approx(0.0) );

  find_bounds_with_interpolation(arr, sizeof(arr)/sizeof(arr[0]), 50, idx_lower, t);
  REQUIRE( idx_lower == 4 );
  REQUIRE( t == Approx(0.0) );

  find_bounds_with_interpolation(arr, sizeof(arr)/sizeof(arr[0]), 5, idx_lower, t);
  REQUIRE( idx_lower == 0 );
  REQUIRE( t == Approx(0.0) );

  find_bounds_with_interpolation(arr, sizeof(arr)/sizeof(arr[0]), 55, idx_lower, t);
  REQUIRE( idx_lower == 4 );
  REQUIRE( t == Approx(0.0) );
}
