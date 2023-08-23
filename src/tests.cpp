#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

int main( int i_argc, char* i_argv[] ) {
  // run unit tests
  int l_result = Catch::Session().run( i_argc, i_argv );

  // return result
  return ( l_result < 0xff ? l_result : 0xff );
}