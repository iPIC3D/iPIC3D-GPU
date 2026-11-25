cmake_minimum_required(VERSION 3.21) # HIP requires 3.21
project(gmmTest LANGUAGES CXX)

# Unit test of the GPU Gaussian-mixture fit (dataAnalysis/GMM) on weighted
# points of a synthetic 3D histogram; no MPI, no field solver.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(HIP_ON "Use HIP" OFF)
set(CUDA_ARCH "75" CACHE STRING "CUDA compute capability to compile for")
set(HIP_ARCH "gfx90a" CACHE STRING "HIP GPU architecture")

if(NOT HIP_ON)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  find_package(HIP REQUIRED)
  enable_language(HIP)
endif()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(INCLUDE_CODE_DIR ${PROJECT_ROOT}/src/include)

set(src_files ${CMAKE_CURRENT_SOURCE_DIR}/gmmTest.cu)
if(HIP_ON)
  set_source_files_properties(${src_files} PROPERTIES LANGUAGE HIP)
endif()

add_executable(gmmTest ${src_files})

# the GMM headers, and Profiler.h of src/include (its regions compile to
# nothing here: no profiler option is set)
set(include_path ${INCLUDE_CODE_DIR}/CUDA ${INCLUDE_CODE_DIR}/dataAnalysis/GMM
                 ${INCLUDE_CODE_DIR})
if(NOT HIP_ON)
  list(APPEND include_path ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  list(APPEND include_path ${INCLUDE_CODE_DIR}/hip)
endif()
target_include_directories(gmmTest PUBLIC ${include_path})

if(NOT HIP_ON)
  set_property(TARGET gmmTest PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
  target_link_libraries(gmmTest PRIVATE hip::device hip::host)
  set_property(TARGET gmmTest PROPERTY HIP_ARCHITECTURES ${HIP_ARCH})
  target_compile_definitions(gmmTest PUBLIC HIPIFLY)
endif()

enable_testing()

# all the cases in one run: they share one GMM object, as the analysis does
include(${CMAKE_CURRENT_SOURCE_DIR}/../unitTestLaunch.cmake)
add_unit_test(gmm TARGET gmmTest TIMEOUT 300 GPU)
