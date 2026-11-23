cmake_minimum_required(VERSION 3.21) # HIP requires 3.21
project(divbCleaningTest LANGUAGES CXX)

# Unit tests of the divergence cleanings (operators, Chebyshev preconditioner
# and its steps, projections, null-space kernels) and of the B absorbing layer
# on the GPU kernels (GPUPhysicsKernels, GPUStencils, GPUBlas, GPUChebyshev,
# GPUCleaning); no MPI, no field solver.

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
set(SOURCE_CODE_DIR ${PROJECT_ROOT}/src/ipic3d)
set(INCLUDE_CODE_DIR ${PROJECT_ROOT}/src/include)

set(src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/divbCleaningTest.cu
    ${SOURCE_CODE_DIR}/cudaKernel/GPUPhysicsKernels.cu
    ${SOURCE_CODE_DIR}/cudaKernel/GPUFieldBoundary.cu
    ${SOURCE_CODE_DIR}/cudaKernel/GPUStencils.cu
    ${SOURCE_CODE_DIR}/cudaKernel/GPUBlas.cu
    ${SOURCE_CODE_DIR}/cudaKernel/GPUChebyshev.cu
    ${SOURCE_CODE_DIR}/cudaKernel/GPUCleaning.cu
    ${SOURCE_CODE_DIR}/cudaKernel/GPUFieldArray.cu)
if(HIP_ON)
  set_source_files_properties(${src_files} PROPERTIES LANGUAGE HIP)
endif()

add_executable(divbCleaningTest ${src_files})

set(include_path ${INCLUDE_CODE_DIR} ${INCLUDE_CODE_DIR}/CUDA)
if(NOT HIP_ON)
  list(APPEND include_path ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  list(APPEND include_path ${INCLUDE_CODE_DIR}/hip)
endif()
target_include_directories(divbCleaningTest PUBLIC ${include_path})

if(NOT HIP_ON)
  set_property(TARGET divbCleaningTest PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
  target_link_libraries(divbCleaningTest PRIVATE hip::device hip::host)
  set_property(TARGET divbCleaningTest PROPERTY HIP_ARCHITECTURES ${HIP_ARCH})
  target_compile_definitions(divbCleaningTest PUBLIC HIPIFLY)
endif()

enable_testing()

include(${CMAKE_CURRENT_SOURCE_DIR}/../unitTestLaunch.cmake)
add_unit_test(cleaning TARGET divbCleaningTest TIMEOUT 120 GPU)
