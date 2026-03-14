#include "catch.hpp"
#include "model_m4.h"

TEST_CASE( "Test full size of gflops_table (m4)" , "[m4]" ) {
    using namespace einsum_ir::model::m4;

    double gflops;

    gflops = gflops_table[M_SIZE-1][N_SIZE-1][K_SIZE-1][1];
    REQUIRE(gflops > 0.0);
}

TEST_CASE( "Find bounds for (M & N) ", "[m4]" ) {
    using namespace einsum_ir::model::m4;

    int idx_lower;
    double t;

    // Exact match
    find_bounds_mn(M_VALUES, M_SIZE, 128, idx_lower, t);
    REQUIRE(idx_lower == 23);
    REQUIRE(t == Approx(0.0));

    // Inside range
    find_bounds_mn(M_VALUES, M_SIZE, 20, idx_lower, t);
    REQUIRE(idx_lower == 3);
    REQUIRE(t == Approx(3.0/14.0));

    // Above range
    find_bounds_mn(M_VALUES, M_SIZE, 300, idx_lower, t);
    REQUIRE(idx_lower == M_SIZE - 3);
    REQUIRE(t == Approx(11.0/14.0));

    find_bounds_mn(M_VALUES, M_SIZE, 230, idx_lower, t);
    REQUIRE(idx_lower == M_SIZE - 6);
    REQUIRE(t == Approx(5.0 / 14.0));

    find_bounds_mn(M_VALUES, M_SIZE, 270, idx_lower, t);
    REQUIRE(idx_lower == M_SIZE - 3);
    REQUIRE(t == Approx(13.0/14.0));
}