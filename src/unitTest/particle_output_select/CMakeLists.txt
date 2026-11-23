cmake_minimum_required(VERSION 3.21) # HIP requires 3.21
project(particleOutputSelectTest LANGUAGES CXX)

# The selection of the particles that the particle output writes
# (particleOutputSelect.cuh) against a CPU reference.

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

set(src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/particleOutputSelectTest.cu
    ${PROJECT_ROOT}/src/ipic3d/cudaKernel/particleOutputSelect.cu)
if(HIP_ON)
  set_source_files_properties(${src_files} PROPERTIES LANGUAGE HIP)
endif()

add_executable(particleOutputSelectTest ${src_files})
set(include_path ${INCLUDE_CODE_DIR} ${INCLUDE_CODE_DIR}/CUDA
                 ${INCLUDE_CODE_DIR}/dataAnalysis)
if(NOT HIP_ON)
  list(APPEND include_path ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  list(APPEND include_path ${INCLUDE_CODE_DIR}/hip)
endif()
target_include_directories(particleOutputSelectTest PUBLIC ${include_path})

if(NOT HIP_ON)
  set_property(TARGET particleOutputSelectTest PROPERTY CUDA_ARCHITECTURES
                                                        ${CUDA_ARCH})
else()
  target_link_libraries(particleOutputSelectTest PRIVATE hip::device hip::host)
  set_property(TARGET particleOutputSelectTest PROPERTY HIP_ARCHITECTURES
                                                        ${HIP_ARCH})
  target_compile_definitions(particleOutputSelectTest PUBLIC HIPIFLY)
endif()

enable_testing()
include(${CMAKE_CURRENT_SOURCE_DIR}/../unitTestLaunch.cmake)
add_unit_test(particle_output_select TARGET particleOutputSelectTest TIMEOUT 120 GPU)
