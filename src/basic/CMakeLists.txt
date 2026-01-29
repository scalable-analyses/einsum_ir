cmake_minimum_required(VERSION 3.21)

project(einsum_ir LANGUAGES CXX VERSION 0.1.0)

# ──────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────
option(EINSUM_IR_ENABLE_TPP           "Enable LIBXSMM TPP backend"              ON)
option(EINSUM_IR_AUTO_INSTALL_LIBXSMM "Auto-install LIBXSMM if not found"       ON)
option(EINSUM_IR_BUNDLE_DEPENDENCIES  "Bundle dependencies for wheel packaging" OFF)
option(BUILD_SHARED_LIBS              "Build shared libraries"                  ON)

# Threading backend selection
set(EINSUM_IR_THREADING_BACKEND "AUTO" CACHE STRING
    "Threading backend: AUTO (auto-detect), DISPATCH (Apple GCD), OPENMP, SEQUENTIAL")
set_property(CACHE EINSUM_IR_THREADING_BACKEND PROPERTY STRINGS AUTO DISPATCH OPENMP SEQUENTIAL)

set(LIBXSMM_GIT_TAG "main" CACHE STRING "Git tag for auto-installed LIBXSMM")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────────────
# LIBXSMM (TPP backend)
# ──────────────────────────────────────────────────────
if(EINSUM_IR_ENABLE_TPP)
  # Validate configuration
  if(LIBXSMM_ROOT AND EINSUM_IR_AUTO_INSTALL_LIBXSMM)
    message(FATAL_ERROR "Cannot use both LIBXSMM_ROOT and EINSUM_IR_AUTO_INSTALL_LIBXSMM. Please set only one.")
  endif()

  set(LIBXSMM_FOUND FALSE)
  set(LIBXSMM_AUTO_INSTALLED FALSE)

  # 1. Manual fallback if user passes -DLIBXSMM_ROOT=<prefix>
  if(LIBXSMM_ROOT)
    message(STATUS "Looking for LIBXSMM in ${LIBXSMM_ROOT}")
    find_path(LIBXSMM_INCLUDE_DIR libxsmm.h PATHS ${LIBXSMM_ROOT}/include NO_DEFAULT_PATH)
    find_library(LIBXSMM_LIBRARY NAMES libxsmm.a
                 PATHS ${LIBXSMM_ROOT}/lib
                 NO_DEFAULT_PATH)
    if(LIBXSMM_INCLUDE_DIR AND LIBXSMM_LIBRARY)
      add_library(LIBXSMM::LIBXSMM STATIC IMPORTED)
      set_target_properties(LIBXSMM::LIBXSMM PROPERTIES
                            IMPORTED_LOCATION ${LIBXSMM_LIBRARY})
      set(LIBXSMM_FOUND TRUE)
      message(STATUS "Found LIBXSMM: ${LIBXSMM_LIBRARY}")
    else()
      message(FATAL_ERROR "LIBXSMM not found in ${LIBXSMM_ROOT}. Please check the path or disable EINSUM_IR_ENABLE_TPP.")
    endif()
  
  # 2. Auto-installation
  elseif(EINSUM_IR_AUTO_INSTALL_LIBXSMM)
    message(STATUS "Auto-installing LIBXSMM from git tag: ${LIBXSMM_GIT_TAG}")
    
    include(ExternalProject)
    
    # Set up build environment
    set(LIBXSMM_PREFIX ${CMAKE_BINARY_DIR}/external/libxsmm)
    set(LIBXSMM_INSTALL_DIR ${LIBXSMM_PREFIX}/install)
    
    # Configure build environment variables
    set(LIBXSMM_BUILD_ENV "")
    if(CMAKE_C_COMPILER)
      list(APPEND LIBXSMM_BUILD_ENV "CC=${CMAKE_C_COMPILER}")
    endif()
    if(CMAKE_CXX_COMPILER)
      list(APPEND LIBXSMM_BUILD_ENV "CXX=${CMAKE_CXX_COMPILER}")
    endif()
    
    # Determine parallel build jobs
    if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
      include(ProcessorCount)
      ProcessorCount(CMAKE_BUILD_PARALLEL_LEVEL)
      if(CMAKE_BUILD_PARALLEL_LEVEL EQUAL 0)
        set(CMAKE_BUILD_PARALLEL_LEVEL 1)
      endif()
    endif()

    # Set variables for later use
    set(LIBXSMM_LIBRARY ${LIBXSMM_INSTALL_DIR}/lib/libxsmm.a)
    set(LIBXSMM_INCLUDE_DIR ${LIBXSMM_INSTALL_DIR}/include)
    set(LIBXSMM_FOUND TRUE)
    set(LIBXSMM_AUTO_INSTALLED TRUE)

    ExternalProject_Add(libxsmm_external
      GIT_REPOSITORY https://github.com/libxsmm/libxsmm.git
      GIT_TAG ${LIBXSMM_GIT_TAG}
      PREFIX ${LIBXSMM_PREFIX}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ${CMAKE_COMMAND} -E env ${LIBXSMM_BUILD_ENV}
                    make BLAS=0 PREFIX=${LIBXSMM_INSTALL_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} install
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND ""
      LOG_DOWNLOAD ON
      LOG_BUILD ON
      BUILD_BYPRODUCTS "${LIBXSMM_LIBRARY}"
    )

    # Create an imported target for the auto-installed LIBXSMM
    add_library(LIBXSMM::LIBXSMM STATIC IMPORTED GLOBAL)
    set_target_properties(LIBXSMM::LIBXSMM PROPERTIES IMPORTED_LOCATION ${LIBXSMM_LIBRARY})
    add_dependencies(LIBXSMM::LIBXSMM libxsmm_external)
  else()
    message(FATAL_ERROR "EINSUM_IR_ENABLE_TPP is ON but LIBXSMM_ROOT is not set and EINSUM_IR_AUTO_INSTALL_LIBXSMM is OFF. Please set LIBXSMM_ROOT or enable EINSUM_IR_AUTO_INSTALL_LIBXSMM.")
  endif()
endif()

# ──────────────────────────────────────────────────────
# Threading Backend Configuration
# ──────────────────────────────────────────────────────

# Auto-detect threading backend
if(EINSUM_IR_THREADING_BACKEND STREQUAL "AUTO")
  if(APPLE)
    set(EINSUM_IR_THREADING_BACKEND_RESOLVED "DISPATCH")
    message(STATUS "Auto-detected threading backend: DISPATCH (macOS)")
  else()
    set(EINSUM_IR_THREADING_BACKEND_RESOLVED "OPENMP")
    message(STATUS "Auto-detected threading backend: OPENMP")
  endif()
else()
  set(EINSUM_IR_THREADING_BACKEND_RESOLVED ${EINSUM_IR_THREADING_BACKEND})
  message(STATUS "Using threading backend: ${EINSUM_IR_THREADING_BACKEND_RESOLVED}")
endif()

# Configure selected threading backend
if(EINSUM_IR_THREADING_BACKEND_RESOLVED STREQUAL "DISPATCH")
  if(NOT APPLE)
    message(FATAL_ERROR "DISPATCH threading backend is only available on macOS")
  endif()
  set(EINSUM_IR_USE_DISPATCH TRUE)
  message(STATUS "Configuring Apple Dispatch (GCD) threading")

elseif(EINSUM_IR_THREADING_BACKEND_RESOLVED STREQUAL "OPENMP")
  find_package(OpenMP REQUIRED)
  set(EINSUM_IR_USE_OPENMP TRUE)
  message(STATUS "Configuring OpenMP threading")

elseif(EINSUM_IR_THREADING_BACKEND_RESOLVED STREQUAL "SEQUENTIAL")
  set(EINSUM_IR_USE_SEQUENTIAL TRUE)
  message(STATUS "Configuring sequential (no threading)")

else()
  message(FATAL_ERROR "Invalid threading backend: ${EINSUM_IR_THREADING_BACKEND_RESOLVED}")
endif()

# ──────────────────────────────────────────────────────
# Sources & target
# ──────────────────────────────────────────────────────
set(src
  binary/ContractionBackend.cpp
  binary/ContractionBackendScalar.cpp
  binary/ContractionOptimizer.cpp
  binary/IterationSpace.cpp
  binary/ContractionMemoryManager.cpp
  unary/UnaryBackend.cpp
  unary/UnaryBackendScalar.cpp
  unary/UnaryOptimizer.cpp)
if(EINSUM_IR_ENABLE_TPP)
  list(APPEND src binary/ContractionBackendTpp.cpp)
  list(APPEND src unary/UnaryBackendTpp.cpp)
endif()

add_library(einsum_ir ${src})
set_target_properties(einsum_ir PROPERTIES INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH TRUE)

# Propagate options and definitions
if(EINSUM_IR_ENABLE_TPP)
    target_compile_definitions(einsum_ir PUBLIC EINSUM_IR_ENABLE_TPP)
endif()

# Threading backend definitions
if(EINSUM_IR_USE_DISPATCH)
  target_compile_definitions(einsum_ir PUBLIC EINSUM_IR_USE_DISPATCH)
elseif(EINSUM_IR_USE_OPENMP)
  target_compile_definitions(einsum_ir PUBLIC EINSUM_IR_USE_OPENMP)
endif()

# Enable position independent code
set_property(TARGET einsum_ir PROPERTY POSITION_INDEPENDENT_CODE ON)

# Link with threading backend
if(EINSUM_IR_USE_DISPATCH)
  # Link with Apple Dispatch framework (part of libSystem, no explicit linking needed)
  # However, we ensure C++ blocks support is available
  target_compile_options(einsum_ir PRIVATE -fblocks)

elseif(EINSUM_IR_USE_OPENMP)
  target_link_libraries(einsum_ir PUBLIC OpenMP::OpenMP_CXX)

  # Special handling for Apple Clang which might need libomp
  if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    target_link_options(einsum_ir PRIVATE ${OpenMP_CXX_FLAGS})
  endif()
endif()

# Link with LIBXSMM if available
if(EINSUM_IR_ENABLE_TPP AND LIBXSMM_FOUND)
  target_link_libraries(einsum_ir PUBLIC LIBXSMM::LIBXSMM)
  target_include_directories(einsum_ir PUBLIC
    $<BUILD_INTERFACE:${LIBXSMM_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

    # Make sure LIBXSMM is built before we compile einsum_ir
    if(LIBXSMM_AUTO_INSTALLED)
        add_dependencies(einsum_ir libxsmm_external)
    endif()
endif()

target_include_directories(einsum_ir
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:include>)

target_compile_features(einsum_ir PUBLIC cxx_std_17)

# ──────────────────────────────────────────────────────
# Install
# ──────────────────────────────────────────────────────
include(GNUInstallDirs)

# Install the einsum_ir library and its dependencies
install(TARGETS einsum_ir EXPORT einsum_irTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Include LIBXSMM headers in the installation
if(EINSUM_IR_ENABLE_TPP AND LIBXSMM_FOUND)
  if(LIBXSMM_AUTO_INSTALLED)
    install(DIRECTORY "${LIBXSMM_INSTALL_DIR}/include/"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")
  else()
    if(LIBXSMM_INCLUDE_DIR)
      install(DIRECTORY "${LIBXSMM_INCLUDE_DIR}/"
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
              FILES_MATCHING PATTERN "*.h")
    endif()
  endif()
endif()

# ──────────────────────────────────────────────────────
# Header Installation
# ──────────────────────────────────────────────────────
set(binary_headers
    binary/ContractionBackend.h
    binary/ContractionBackendScalar.h
    binary/ContractionOptimizer.h
    binary/IterationSpace.h
    binary/ContractionMemoryManager.h)
if(EINSUM_IR_ENABLE_TPP)
  list(APPEND binary_headers binary/ContractionBackendTpp.h)
endif()

set(unary_headers
    unary/UnaryBackend.h
    unary/UnaryBackendScalar.h
    unary/UnaryOptimizer.h)
if(EINSUM_IR_ENABLE_TPP)
  list(APPEND unary_headers unary/UnaryBackendTpp.h)
endif()

set(top_level_headers
  constants.h
  threading.h)

# Install all headers in one consistent block
install(FILES ${binary_headers} 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/einsum_ir/binary)

install(FILES ${unary_headers} 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/einsum_ir/unary)

install(FILES ${top_level_headers} 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/einsum_ir)

install(EXPORT einsum_irTargets
        NAMESPACE einsum_ir::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einsum_ir)

# Generate and install the config file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/einsum_irConfig.cmake"
    "include(CMakeFindDependencyMacro)\n"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/einsum_irTargets.cmake\")\n"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/einsum_irConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einsum_ir)