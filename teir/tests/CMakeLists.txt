# C++ test executable (Catch2 v3, registered with CTest).

set(TEIR_TEST_SOURCES
  walker.test.cpp
)

if(ETOPS_LIBXSMM_AVAILABLE)
  list(APPEND TEIR_TEST_SOURCES tpp_dispatch.test.cpp)
endif()

if(ETOPS_BLAS_AVAILABLE)
  list(APPEND TEIR_TEST_SOURCES blas_dispatch.test.cpp)
endif()

add_executable(teir_tests ${TEIR_TEST_SOURCES})

target_link_libraries(teir_tests PRIVATE teir Catch2::Catch2WithMain)
target_compile_features(teir_tests PRIVATE cxx_std_20)
target_compile_options(teir_tests PRIVATE -Wall -Wextra -Wpedantic)

catch_discover_tests(teir_tests)
