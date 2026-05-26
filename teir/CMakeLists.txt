cmake_minimum_required(VERSION 3.20)

project(teir
  VERSION 0.1.0
  DESCRIPTION "Tiled Execution Intermediate Representation runtime"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# ----------------------------------------------------------------------------
# User-facing options
# ----------------------------------------------------------------------------
option(ETOPS_ENABLE_TESTS "Build C++ tests"             OFF)
option(ETOPS_ENABLE_SANITIZERS "Enable ASan/UBSan"      OFF)
option(ETOPS_USE_SYSTEM_LIBXSMM "Use system libxsmm"    OFF)
option(ETOPS_BUILD_PYTHON "Build the pybind11 module"   ON)

set(ETOPS_THREADING_BACKEND "AUTO" CACHE STRING
    "Threading backend: AUTO | DISPATCH | OPENMP | SEQUENTIAL")
set_property(CACHE ETOPS_THREADING_BACKEND PROPERTY STRINGS
             AUTO DISPATCH OPENMP SEQUENTIAL)

# ----------------------------------------------------------------------------
# Threading backend resolution
# ----------------------------------------------------------------------------
if(ETOPS_THREADING_BACKEND STREQUAL "AUTO")
  if(APPLE)
    set(ETOPS_THREADING_BACKEND_RESOLVED "DISPATCH")
  else()
    set(ETOPS_THREADING_BACKEND_RESOLVED "OPENMP")
  endif()
else()
  set(ETOPS_THREADING_BACKEND_RESOLVED "${ETOPS_THREADING_BACKEND}")
endif()

message(STATUS "Threading backend: ${ETOPS_THREADING_BACKEND_RESOLVED}")

if(ETOPS_THREADING_BACKEND_RESOLVED STREQUAL "DISPATCH")
  if(NOT APPLE)
    message(FATAL_ERROR "DISPATCH threading backend is only supported on Apple platforms.")
  endif()
  set(ETOPS_THREADING_BACKEND_ID 1)
elseif(ETOPS_THREADING_BACKEND_RESOLVED STREQUAL "OPENMP")
  find_package(OpenMP REQUIRED)
  set(ETOPS_THREADING_BACKEND_ID 2)
elseif(ETOPS_THREADING_BACKEND_RESOLVED STREQUAL "SEQUENTIAL")
  set(ETOPS_THREADING_BACKEND_ID 3)
else()
  message(FATAL_ERROR "Unknown threading backend: ${ETOPS_THREADING_BACKEND_RESOLVED}")
endif()

# ----------------------------------------------------------------------------
# Sanitizers (debug only)
# ----------------------------------------------------------------------------
set(ETOPS_SANITIZERS "" CACHE STRING
    "Comma-separated list of sanitizers (address, undefined, thread). \
Empty disables sanitizers.")

# Backwards compatibility: ETOPS_ENABLE_SANITIZERS=ON selects ASan + UBSan.
if(ETOPS_ENABLE_SANITIZERS)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Sanitizers requested but build type is ${CMAKE_BUILD_TYPE}.")
  endif()
  if("${ETOPS_SANITIZERS}" STREQUAL "")
    set(ETOPS_SANITIZERS "address,undefined")
  endif()
endif()

if(NOT "${ETOPS_SANITIZERS}" STREQUAL "")
  message(STATUS "Enabling sanitizers: ${ETOPS_SANITIZERS}")
  add_compile_options(
    -fsanitize=${ETOPS_SANITIZERS}
    -fno-omit-frame-pointer
    -fno-sanitize-recover=all
  )
  add_link_options(
    -fsanitize=${ETOPS_SANITIZERS}
  )
  if(NOT ETOPS_THREADING_BACKEND_RESOLVED STREQUAL "SEQUENTIAL")
    message(WARNING
      "Sanitizer builds default to SEQUENTIAL threading; "
      "current backend is ${ETOPS_THREADING_BACKEND_RESOLVED}.")
  endif()
endif()

# ----------------------------------------------------------------------------
# libxsmm
# ----------------------------------------------------------------------------
include(FetchContent)

if(ETOPS_USE_SYSTEM_LIBXSMM OR DEFINED ENV{LIBXSMM_ROOT} OR DEFINED LIBXSMM_ROOT)
  if(NOT DEFINED LIBXSMM_ROOT AND DEFINED ENV{LIBXSMM_ROOT})
    set(LIBXSMM_ROOT $ENV{LIBXSMM_ROOT})
  endif()
  if(DEFINED LIBXSMM_ROOT)
    message(STATUS "Using libxsmm at ${LIBXSMM_ROOT}")
    set(ETOPS_LIBXSMM_AVAILABLE ON)
    set(ETOPS_LIBXSMM_INCLUDE_DIR "${LIBXSMM_ROOT}/include")
    set(ETOPS_LIBXSMM_LIBRARY_DIR "${LIBXSMM_ROOT}/lib")
  else()
    message(STATUS "ETOPS_USE_SYSTEM_LIBXSMM=ON but no LIBXSMM_ROOT; trying find_package.")
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
      pkg_check_modules(LIBXSMM libxsmm)
    endif()
    if(LIBXSMM_FOUND)
      set(ETOPS_LIBXSMM_AVAILABLE ON)
    else()
      set(ETOPS_LIBXSMM_AVAILABLE OFF)
      message(WARNING "libxsmm not found; TPP backend disabled.")
    endif()
  endif()
else()
  set(ETOPS_LIBXSMM_AVAILABLE ON)
  # Pinned libxsmm commit. Update intentionally and re-run the C++ + Python
  # test suites
  set(LIBXSMM_GIT_TAG
      "c14cbc6f8bc7964f8c5190a3a16b8cace03e5889"
      CACHE STRING "libxsmm git ref (pinned to a tested commit by default)")
  message(STATUS "Fetching libxsmm @ ${LIBXSMM_GIT_TAG}")
  FetchContent_Declare(libxsmm_src
    GIT_REPOSITORY https://github.com/libxsmm/libxsmm.git
    GIT_TAG        ${LIBXSMM_GIT_TAG}
  )
  FetchContent_MakeAvailable(libxsmm_src)
  set(ETOPS_LIBXSMM_INCLUDE_DIR "${libxsmm_src_SOURCE_DIR}/include")
endif()

# ----------------------------------------------------------------------------
# BLAS
# ----------------------------------------------------------------------------
set(BLA_VENDOR "" CACHE STRING "BLAS vendor preference")
set(CBLAS_INCLUDE_DIR "" CACHE PATH
    "Directory containing cblas.h; falls back to find_path probing if unset")
find_package(BLAS)
if(BLAS_FOUND)
  set(ETOPS_BLAS_AVAILABLE ON)
  if(BLAS_LIBRARIES MATCHES "Accelerate" OR BLAS_LIBRARIES MATCHES "accelerate")
    set(ETOPS_BLAS_VENDOR_RESOLVED "ACCELERATE")
    set(ETOPS_BLAS_VENDOR_ID 3)
  elseif(BLAS_LIBRARIES MATCHES "openblas" OR BLAS_LIBRARIES MATCHES "OpenBLAS")
    set(ETOPS_BLAS_VENDOR_RESOLVED "OPENBLAS")
    set(ETOPS_BLAS_VENDOR_ID 1)
  elseif(BLAS_LIBRARIES MATCHES "mkl" OR BLAS_LIBRARIES MATCHES "MKL")
    set(ETOPS_BLAS_VENDOR_RESOLVED "MKL")
    set(ETOPS_BLAS_VENDOR_ID 2)
  else()
    set(ETOPS_BLAS_VENDOR_RESOLVED "GENERIC")
    set(ETOPS_BLAS_VENDOR_ID 4)
  endif()
  message(STATUS "BLAS vendor: ${ETOPS_BLAS_VENDOR_RESOLVED} (${BLAS_LIBRARIES})")

  # Locate cblas.h. Apple Accelerate exposes it via framework include
  # paths handled by find_package; for every other vendor we either
  # respect a user-supplied CBLAS_INCLUDE_DIR / OPENBLAS_ROOT or fall
  # back to a find_path probe that walks the same hints find_package
  # used to locate the library.
  if(NOT ETOPS_BLAS_VENDOR_RESOLVED STREQUAL "ACCELERATE")
    set(_blas_lib "")
    list(LENGTH BLAS_LIBRARIES _blas_n)
    if(_blas_n GREATER 0)
      list(GET BLAS_LIBRARIES 0 _blas_lib)
    endif()
    set(_blas_libdir "")
    if(_blas_lib)
      get_filename_component(_blas_libdir "${_blas_lib}" DIRECTORY)
    endif()
    if(CBLAS_INCLUDE_DIR STREQUAL "")
      find_path(CBLAS_INCLUDE_DIR_FOUND
        NAMES cblas.h
        HINTS
          ${OPENBLAS_ROOT}/include
          $ENV{OPENBLAS_ROOT}/include
          $ENV{BLAS_ROOT}/include
          ${_blas_libdir}/../include
          ${_blas_libdir}/../../include
          /usr/include/openblas
          /usr/include
          /usr/local/include
        DOC "Directory containing cblas.h")
      if(CBLAS_INCLUDE_DIR_FOUND)
        set(CBLAS_INCLUDE_DIR "${CBLAS_INCLUDE_DIR_FOUND}" CACHE PATH "" FORCE)
      else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
          pkg_check_modules(_OPENBLAS QUIET openblas)
          if(_OPENBLAS_FOUND)
            list(GET _OPENBLAS_INCLUDE_DIRS 0 CBLAS_INCLUDE_DIR_PKG)
            if(CBLAS_INCLUDE_DIR_PKG)
              set(CBLAS_INCLUDE_DIR "${CBLAS_INCLUDE_DIR_PKG}" CACHE PATH "" FORCE)
            endif()
          endif()
        endif()
      endif()
    endif()
    if(CBLAS_INCLUDE_DIR)
      message(STATUS "cblas header: ${CBLAS_INCLUDE_DIR}")
    else()
      message(WARNING
        "BLAS library found at ${BLAS_LIBRARIES} but cblas.h not located. "
        "Set -DCBLAS_INCLUDE_DIR=<path> to point at the directory that "
        "contains cblas.h, or unset to disable the BLAS backend.")
      set(ETOPS_BLAS_AVAILABLE OFF)
      set(ETOPS_BLAS_VENDOR_RESOLVED "UNAVAILABLE")
      set(ETOPS_BLAS_VENDOR_ID 0)
    endif()
  endif()
else()
  set(ETOPS_BLAS_AVAILABLE OFF)
  set(ETOPS_BLAS_VENDOR_RESOLVED "UNAVAILABLE")
  set(ETOPS_BLAS_VENDOR_ID 0)
  message(WARNING "BLAS not found; BLAS backend disabled.")
endif()

# ----------------------------------------------------------------------------
# Configure-time header
# ----------------------------------------------------------------------------
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/include/teir/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/teir/config.h"
  @ONLY
)

# ----------------------------------------------------------------------------
# libteir static library
# ----------------------------------------------------------------------------
set(TEIR_SOURCES
  src/runtime/walker.cpp
  src/runtime/threading.cpp
  src/primitives/registry.cpp
)

add_library(teir STATIC ${TEIR_SOURCES})
target_include_directories(teir
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
)
target_compile_features(teir PUBLIC cxx_std_20)

if(ETOPS_THREADING_BACKEND_RESOLVED STREQUAL "OPENMP")
  target_link_libraries(teir PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(teir PRIVATE
  -Wall -Wextra -Wpedantic
  -Wno-unused-parameter
)

# TPP backend
if(ETOPS_LIBXSMM_AVAILABLE)
  target_sources(teir PRIVATE
    src/backends/tpp/lowering.cpp
  )
  if(TARGET xsmm)
    # FetchContent-built libxsmm exports an `xsmm` target.
    target_link_libraries(teir PRIVATE xsmm)
    target_include_directories(teir PRIVATE
      $<TARGET_PROPERTY:xsmm,INCLUDE_DIRECTORIES>
      "${libxsmm_src_SOURCE_DIR}/src/template"
    )
  elseif(TARGET libxsmm)
    target_link_libraries(teir PRIVATE libxsmm)
  elseif(ETOPS_LIBXSMM_LIBRARY_DIR)
    target_include_directories(teir PRIVATE ${ETOPS_LIBXSMM_INCLUDE_DIR})
    target_link_directories(teir PRIVATE ${ETOPS_LIBXSMM_LIBRARY_DIR})
    target_link_libraries(teir PRIVATE xsmm)
  endif()
endif()

# BLAS backend
if(ETOPS_BLAS_AVAILABLE)
  target_sources(teir PRIVATE
    src/backends/blas/lowering.cpp
  )
  if(NOT ETOPS_BLAS_VENDOR_RESOLVED STREQUAL "ACCELERATE" AND CBLAS_INCLUDE_DIR)
    target_include_directories(teir PRIVATE ${CBLAS_INCLUDE_DIR})
  endif()
  target_link_libraries(teir PRIVATE ${BLAS_LIBRARIES})
endif()

# ----------------------------------------------------------------------------
# Python bindings
# ----------------------------------------------------------------------------
if(ETOPS_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native MODULE
    bindings/python/module.cpp
    bindings/python/convert.cpp
  )
  target_link_libraries(_native PRIVATE teir)
  target_include_directories(_native PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
  install(TARGETS _native DESTINATION etops)
endif()

# ----------------------------------------------------------------------------
# C++ tests (off by default; on for the standalone CMake workflow)
# ----------------------------------------------------------------------------
if(ETOPS_ENABLE_TESTS)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  include(Catch)
  enable_testing()
  add_subdirectory(tests)
endif()
