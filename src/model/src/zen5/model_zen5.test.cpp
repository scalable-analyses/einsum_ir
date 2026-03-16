#include "catch.hpp"
#include "model_zen5.h"

TEST_CASE( "Test full size of gflops_table (zen5)" , "[zen5]" ) {
    using namespace einsum_ir::model::zen5;

    double gflops;

    gflops = gflops_table[M_SIZE-1][N_SIZE-1][K_SIZE-1][1][1];
    REQUIRE(gflops > 0.0);
}

TEST_CASE( "Find bounds for M ", "[zen5]" ) {
    using namespace einsum_ir::model::zen5;

    int idx_lower;
    double t;

    // Exact match
    find_bounds_m(M_VALUES, M_SIZE, 64, idx_lower, t);
    REQUIRE(idx_lower == 11);
    REQUIRE(t == Approx(0.0));

    // Below range
    find_bounds_m(M_VALUES, M_SIZE, 8, idx_lower, t);
    REQUIRE(idx_lower == 0);
    REQUIRE(t == Approx(0.5));

    // Above range
    find_bounds_m(M_VALUES, M_SIZE, 200, idx_lower, t);
    REQUIRE(idx_lower == M_SIZE - 3);
    REQUIRE(t == Approx(0.5));

    find_bounds_m(M_VALUES, M_SIZE, 100, idx_lower, t);
    REQUIRE(idx_lower == 18);
    REQUIRE(t == Approx(3.0 / 14.0));

    find_bounds_m(M_VALUES, M_SIZE, 145, idx_lower, t);
    REQUIRE(idx_lower == M_SIZE - 3);
    REQUIRE(t == Approx(0.0));
}