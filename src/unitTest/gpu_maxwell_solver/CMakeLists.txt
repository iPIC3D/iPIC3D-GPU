cmake_minimum_required(VERSION 3.21)
project(gpuMaxwellSolverTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(HIP_ON "Use HIP" OFF)
option(HALO_OVERLAP "Compile GPU field code with split halo overlap" OFF)
set(CUDA_ARCH "75" CACHE STRING "CUDA compute capability to compile for")
set(HIP_ARCH "gfx90a" CACHE STRING "HIP GPU architecture")
# Keep this default in sync with TestInput::{x,y,z}len.
set(GPU_MAXWELL_MPI_PROCS "8" CACHE STRING "MPI ranks used by CTest")

if(NOT HIP_ON)
  enable_language(CUDA)
  message(STATUS "Using CUDA environment.")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
else()
  find_package(HIP REQUIRED)
  enable_language(HIP)
  message(STATUS "Using HIP environment.")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(SOURCE_CODE_DIR ${PROJECT_ROOT}/src/ipic3d)
set(INCLUDE_CODE_DIR ${PROJECT_ROOT}/src/include)

file(
  GLOB
  src_files
  ${SOURCE_CODE_DIR}/ConfigFile/src/*.cpp
  ${SOURCE_CODE_DIR}/PSKOutput3D/*.cpp
  ${SOURCE_CODE_DIR}/bc/*.cpp
  ${SOURCE_CODE_DIR}/communication/*.cpp
  ${SOURCE_CODE_DIR}/fields/*.cpp
  ${SOURCE_CODE_DIR}/grids/*.cpp
  ${SOURCE_CODE_DIR}/inputoutput/*.cpp
  ${SOURCE_CODE_DIR}/mathlib/*.cpp
  ${SOURCE_CODE_DIR}/particles/*.cpp
  ${SOURCE_CODE_DIR}/performances/*.cpp
  ${SOURCE_CODE_DIR}/solvers/*.cpp
  ${SOURCE_CODE_DIR}/utility/*.cpp
  ${SOURCE_CODE_DIR}/main/*.cpp
)

file(GLOB cuda_src_files
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
     ${SOURCE_CODE_DIR}/cudaKernel/*.cu
     ${SOURCE_CODE_DIR}/main/*.cu
     ${SOURCE_CODE_DIR}/dataAnalysis/*.cu)
if(HIP_ON)
  set_source_files_properties(${cuda_src_files} PROPERTIES LANGUAGE HIP)
endif()
list(APPEND src_files ${cuda_src_files})

add_executable(gpuMaxwellSolverTest ${src_files})

set(include_path ${INCLUDE_CODE_DIR})
list(APPEND include_path ${INCLUDE_CODE_DIR}/CUDA)
list(APPEND include_path ${INCLUDE_CODE_DIR}/dataAnalysis)
list(APPEND include_path ${INCLUDE_CODE_DIR}/dataAnalysis/GMM)
list(APPEND include_path ${INCLUDE_CODE_DIR}/dataAnalysis/Histogram)
if(NOT HIP_ON)
  list(APPEND include_path ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  list(APPEND include_path ${INCLUDE_CODE_DIR}/hip)
endif()

target_include_directories(gpuMaxwellSolverTest PUBLIC ${include_path})
target_link_libraries(gpuMaxwellSolverTest PRIVATE MPI::MPI_CXX)
target_compile_definitions(gpuMaxwellSolverTest PUBLIC GPU_SOLVER NO_HDF5)
if(HALO_OVERLAP)
  target_compile_definitions(gpuMaxwellSolverTest PUBLIC HALO_OVERLAP)
endif()

if(NOT HIP_ON)
  set_target_properties(gpuMaxwellSolverTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET gpuMaxwellSolverTest PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
  set_property(TARGET gpuMaxwellSolverTest PROPERTY CUDA_STANDARD 17)
  set_property(TARGET gpuMaxwellSolverTest PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET gpuMaxwellSolverTest PROPERTY LINK_OPTIONS -fno-lto)
else()
  target_link_libraries(gpuMaxwellSolverTest PRIVATE hip::device hip::host)
  target_compile_options(gpuMaxwellSolverTest PUBLIC -fgpu-rdc)
  target_link_options(gpuMaxwellSolverTest PUBLIC -fgpu-rdc --hip-link)
  set_property(TARGET gpuMaxwellSolverTest PROPERTY HIP_ARCHITECTURES ${HIP_ARCH})
  target_compile_definitions(gpuMaxwellSolverTest PUBLIC HIPIFLY)
endif()

include(CTest)
enable_testing()

add_test(
  NAME gpuMaxwellSolverPeriodic
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${GPU_MAXWELL_MPI_PROCS}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:gpuMaxwellSolverTest>
          --case periodic ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  gpuMaxwellSolverPeriodic
  PROPERTIES
    TIMEOUT 90
  )

add_test(
  NAME gpuMaxwellSolverReflective
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${GPU_MAXWELL_MPI_PROCS}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:gpuMaxwellSolverTest>
          --case reflective ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  gpuMaxwellSolverReflective
  PROPERTIES
    TIMEOUT 90
  )

add_test(
  NAME gpuMaxwellSolverManufactured
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${GPU_MAXWELL_MPI_PROCS}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:gpuMaxwellSolverTest>
          --case manufactured ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  gpuMaxwellSolverManufactured
  PROPERTIES
    TIMEOUT 90
  )

add_test(
  NAME gpuMaxwellSolverObliqueWaveConvergence
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${GPU_MAXWELL_MPI_PROCS}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:gpuMaxwellSolverTest>
          --case oblique-wave ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  gpuMaxwellSolverObliqueWaveConvergence
  PROPERTIES
    PROCESSORS ${GPU_MAXWELL_MPI_PROCS}
    RUN_SERIAL TRUE
    TIMEOUT 300
  )
