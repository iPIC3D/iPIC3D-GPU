cmake_minimum_required(VERSION 3.21) # HIP requires 3.21
project(particleControlTest LANGUAGES CXX)

# Unit test of the particle merging and splitting kernels
# (particleControlKernel.cu) on small sets of particles; no field solver.
# MPI is needed for its header only (the particle ID generator).

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

find_package(MPI REQUIRED COMPONENTS CXX)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(INCLUDE_CODE_DIR ${PROJECT_ROOT}/src/include)

set(src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/particleControlTest.cu
    ${PROJECT_ROOT}/src/ipic3d/cudaKernel/particleControlKernel.cu)
if(HIP_ON)
  set_source_files_properties(${src_files} PROPERTIES LANGUAGE HIP)
endif()

add_executable(particleControlTest ${src_files})

set(include_path ${INCLUDE_CODE_DIR} ${INCLUDE_CODE_DIR}/CUDA)
if(NOT HIP_ON)
  list(APPEND include_path ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  list(APPEND include_path ${INCLUDE_CODE_DIR}/hip)
endif()
target_include_directories(particleControlTest PUBLIC ${include_path})
target_link_libraries(particleControlTest PRIVATE MPI::MPI_CXX)

if(NOT HIP_ON)
  set_property(TARGET particleControlTest PROPERTY CUDA_ARCHITECTURES
                                                   ${CUDA_ARCH})
else()
  target_link_libraries(particleControlTest PRIVATE hip::device hip::host)
  set_property(TARGET particleControlTest PROPERTY HIP_ARCHITECTURES
                                                   ${HIP_ARCH})
  target_compile_definitions(particleControlTest PUBLIC HIPIFLY)
endif()

enable_testing()

foreach(test_case merge-conservation merge-host merge-frame merge-chain
                  merge-nodes split split-merge)
  string(REPLACE "-" "_" test_name ${test_case})
  add_test(NAME particleControl_${test_name}
           COMMAND $<TARGET_FILE:particleControlTest> --case ${test_case})
  set_tests_properties(particleControl_${test_name} PROPERTIES TIMEOUT 120)
endforeach()
