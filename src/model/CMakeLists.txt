cmake_minimum_required(VERSION 3.14)
project(model)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

find_library(LIBXSMM_LIBRARY
    NAMES xsmm libxsmm
    PATHS ${CMAKE_SOURCE_DIR}/../../libxsmm/lib
    NO_DEFAULT_PATH
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(LIBRARY_SOURCES
    src/common/common.cpp
    src/common/interpolation.cpp
    src/m4/model_m4.cpp
    src/m4/bench_m4.cpp
    src/zen5/model_zen5.cpp
    src/zen5/bench_zen5.cpp
    src/a76/model_a76.cpp
    src/a76/bench_a76.cpp
    src/generic/model_generic.cpp
)

add_library(perf_model STATIC ${LIBRARY_SOURCES})


if(LIBXSMM_LIBRARY)
    include_directories(${CMAKE_SOURCE_DIR}/../../libxsmm/include)
    
    set(EXECUTABLE_SOURCES
        src/bench_model.cpp
        src/m4/bench_m4.cpp
        src/zen5/bench_zen5.cpp
        src/a76/bench_a76.cpp
    )

    add_executable(bench_model ${EXECUTABLE_SOURCES})
    target_link_libraries(bench_model perf_model ${LIBXSMM_LIBRARY})
endif()

# Testing setup
enable_testing()

# Include Catch2 header location
include_directories(${CMAKE_SOURCE_DIR}/../../)

# Test sources
set(TEST_SOURCES
    src/tests.cpp
    src/common/common.test.cpp
    src/zen5/model_zen5.test.cpp
    src/m4/model_m4.test.cpp
    src/a76/model_a76.test.cpp
)

# Create test executable
add_executable(model_tests ${TEST_SOURCES})
target_link_libraries(model_tests perf_model)

# Add test to CTest
add_test(NAME model_tests COMMAND model_tests)
