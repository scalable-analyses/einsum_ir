#include "catch.hpp"
#include "model_a76.h"

TEST_CASE( "Test full size of gflops_table (a76)", "[a76]") {

    using namespace einsum_ir::model::a76;

    double gflops;

    gflops = gflops_table[M_SIZE-1][N_SIZE-1][K_SIZE-1][1][1];
    REQUIRE(gflops > 0.0);
}

TEST_CASE( "Get index", "[a76]") {

    using namespace einsum_ir::model::a76;

    int idx;

    idx = get_exact_index(M_VALUES, M_SIZE, 10);
    REQUIRE(idx == 9);

    idx = get_exact_index(M_VALUES, M_SIZE, 16);
    REQUIRE(idx == 15);

    idx = get_exact_index(M_VALUES, M_SIZE, 20);
    REQUIRE(idx == 15);

    idx = get_exact_index(N_VALUES, N_SIZE, 5);
    REQUIRE(idx == 4);

    idx = get_exact_index(N_VALUES, N_SIZE, 15);
    REQUIRE(idx == 14);

    idx = get_exact_index(N_VALUES, N_SIZE, 20);
    REQUIRE(idx == 14);
}