cmake_minimum_required(VERSION 3.21)
project(haloCommunicationTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(HIP_ON "Use HIP" OFF)
option(HALO_OVERLAP "Compile GPU field code with split halo overlap" OFF)
set(CUDA_ARCH "75" CACHE STRING "CUDA compute capability to compile for")
set(HIP_ARCH "gfx90a" CACHE STRING "HIP GPU architecture")

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
  GLOB src_files CONFIGURE_DEPENDS
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

file(
  GLOB cuda_src_files CONFIGURE_DEPENDS
  ${SOURCE_CODE_DIR}/cudaKernel/*.cu
  ${SOURCE_CODE_DIR}/main/*.cu
  ${SOURCE_CODE_DIR}/dataAnalysis/*.cu
)
list(APPEND cuda_src_files ${CMAKE_CURRENT_SOURCE_DIR}/haloCommunicationTest.cu)
if(HIP_ON)
  set_source_files_properties(${cuda_src_files} PROPERTIES LANGUAGE HIP)
endif()
list(APPEND src_files ${cuda_src_files})

add_executable(haloCommunicationTest ${src_files})

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

target_include_directories(haloCommunicationTest PUBLIC ${include_path})
target_link_libraries(haloCommunicationTest PRIVATE MPI::MPI_CXX)
target_compile_definitions(haloCommunicationTest PUBLIC GPU_SOLVER NO_HDF5)
if(HALO_OVERLAP)
  target_compile_definitions(haloCommunicationTest PUBLIC HALO_OVERLAP)
endif()

if(NOT HIP_ON)
  set_target_properties(haloCommunicationTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET haloCommunicationTest PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
  set_property(TARGET haloCommunicationTest PROPERTY CUDA_STANDARD 17)
  set_property(TARGET haloCommunicationTest PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET haloCommunicationTest PROPERTY LINK_OPTIONS -fno-lto)
else()
  target_link_libraries(haloCommunicationTest PRIVATE hip::device hip::host)
  target_compile_options(haloCommunicationTest PUBLIC -fgpu-rdc)
  target_link_options(haloCommunicationTest PUBLIC -fgpu-rdc --hip-link)
  set_property(TARGET haloCommunicationTest PROPERTY HIP_ARCHITECTURES ${HIP_ARCH})
  target_compile_definitions(haloCommunicationTest PUBLIC HIPIFLY)
endif()

include(CTest)
enable_testing()

function(add_halo_communication_test name mpi_ranks exchange_case suite backend)
  add_test(
    NAME ${name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${mpi_ranks}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:haloCommunicationTest>
            ${MPIEXEC_POSTFLAGS}
            --case ${exchange_case} --suite ${suite} --backend ${backend})
  set_tests_properties(
    ${name}
    PROPERTIES
      PROCESSORS ${mpi_ranks}
      RUN_SERIAL TRUE
      TIMEOUT 90
  )
  if(backend STREQUAL "gpu")
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endfunction()

# Keep CPU and GPU registrations separate: CPU validation remains available on
# hosts without a device, while GPU-only cases can report CTest's skip code 77.
add_halo_communication_test(
  haloCommunicationConventionalDistributedCPU 8 conventional distributed cpu)
add_halo_communication_test(
  haloCommunicationConventionalDistributedGPU 8 conventional distributed gpu)
add_halo_communication_test(
  haloCommunicationAdditiveDistributedCPU 8 additive distributed cpu)
add_halo_communication_test(
  haloCommunicationAdditiveDistributedGPU 8 additive distributed gpu)
add_halo_communication_test(
  haloCommunicationConventionalSelfCPU 1 conventional self cpu)
add_halo_communication_test(
  haloCommunicationConventionalSelfGPU 1 conventional self gpu)
add_halo_communication_test(
  haloCommunicationAdditiveSelfCPU 1 additive self cpu)
add_halo_communication_test(
  haloCommunicationAdditiveSelfGPU 1 additive self gpu)
