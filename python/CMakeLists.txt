cmake_minimum_required(VERSION 3.20)
project(etops LANGUAGES CXX)

# Check for OpenMP availability
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    set(ETOPS_USE_OPENMP ON)
    set(EINSUM_IR_ENABLE_OPENMP ON)
else()
    set(ETOPS_USE_OPENMP OFF)
    set(EINSUM_IR_ENABLE_OPENMP OFF)
endif()

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(ETOPS_USE_PAPI "Enable Papi profiling" OFF)

# ------------------------------------------------------------
# einsum_ir
# ------------------------------------------------------------
message(STATUS "Building einsum_ir from ../src/basic")
add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../src/basic"
    einsum_ir_local
    EXCLUDE_FROM_ALL
)

# ------------------------------------------------------------
# Copy headers so that <einsum_ir/basic/...> resolves in-tree
# ------------------------------------------------------------
file(
    COPY
    "${CMAKE_CURRENT_LIST_DIR}/../src/basic"
    DESTINATION "${CMAKE_BINARY_DIR}/include/einsum_ir"
)

# ------------------------------------------------------------
# Sources & target
# ------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

add_library(TensorOperation STATIC src/TensorOperation.cpp)
target_link_libraries(TensorOperation PUBLIC einsum_ir)
target_include_directories(
    TensorOperation
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
)

if(ETOPS_USE_OPENMP)
    target_link_libraries(TensorOperation PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(_etops_core MODULE src/bindings.cpp)
target_link_libraries(_etops_core PRIVATE TensorOperation)

if(APPLE)
    set_target_properties(
        _etops_core PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_etops_core"
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
else()
    set_target_properties(
        _etops_core PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_etops_core"
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        LINK_FLAGS "-Wl,--enable-new-dtags"
    )
endif()

if(ETOPS_USE_PAPI)
  include(ExternalProject)

  set(PAPI_PREFIX_DIR "${CMAKE_BINARY_DIR}/external/papi")
  set(PAPI_SOURCE_DIR "${PAPI_PREFIX_DIR}/src/papi_external")
  set(PAPI_INSTALL_DIR "${PAPI_SOURCE_DIR}/src/install")

    file(MAKE_DIRECTORY
        "${PAPI_INSTALL_DIR}/include"
        "${PAPI_INSTALL_DIR}/lib"
    )
  
  ExternalProject_Add(papi_external
    GIT_REPOSITORY https://github.com/icl-utk-edu/papi.git
    GIT_TAG master
        PREFIX ${PAPI_PREFIX_DIR}
        SOURCE_DIR ${PAPI_SOURCE_DIR}
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR>/src ./configure --prefix=<SOURCE_DIR>/src/install
        BUILD_COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR>/src make
        INSTALL_COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR>/src make install
        BUILD_BYPRODUCTS "${PAPI_INSTALL_DIR}/lib/libpapi.a"
  )

    add_library(PAPI::PAPI STATIC IMPORTED GLOBAL)
    set_target_properties(PAPI::PAPI PROPERTIES
    IMPORTED_LOCATION "${PAPI_INSTALL_DIR}/lib/libpapi.a"
        INTERFACE_INCLUDE_DIRECTORIES "${PAPI_INSTALL_DIR}/include"
  )
  add_dependencies(PAPI::PAPI papi_external)

  target_link_libraries(TensorOperation PUBLIC PAPI::PAPI)
  target_compile_definitions(TensorOperation PUBLIC ETOPS_USE_PAPI)
endif()

target_compile_features(_etops_core PUBLIC)

# ------------------------------------------------------------
# Install
# ------------------------------------------------------------
install(
    TARGETS einsum_ir
    LIBRARY DESTINATION etops
    ARCHIVE DESTINATION etops
    RUNTIME DESTINATION etops
)

install(
    TARGETS _etops_core
    LIBRARY DESTINATION etops
    RUNTIME DESTINATION etops
)