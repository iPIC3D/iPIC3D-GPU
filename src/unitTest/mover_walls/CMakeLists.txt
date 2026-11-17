cmake_minimum_required(VERSION 3.21) # HIP requires 3.21
project(moverWallsTest LANGUAGES CXX)

# Unit test of the particle push with reflecting walls (moverPush.cuh): the
# pushes of the mover kernels and pushWithWalls with uniform fields, on the
# host and on the device; no MPI, no field solver.

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

set(src_files ${CMAKE_CURRENT_SOURCE_DIR}/moverWallsTest.cu)
if(HIP_ON)
  set_source_files_properties(${src_files} PROPERTIES LANGUAGE HIP)
endif()

add_executable(moverWallsTest ${src_files})

set(include_path ${INCLUDE_CODE_DIR} ${INCLUDE_CODE_DIR}/CUDA)
if(NOT HIP_ON)
  list(APPEND include_path ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  list(APPEND include_path ${INCLUDE_CODE_DIR}/hip)
endif()
target_include_directories(moverWallsTest PUBLIC ${include_path})

if(NOT HIP_ON)
  set_property(TARGET moverWallsTest PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
  target_link_libraries(moverWallsTest PRIVATE hip::device hip::host)
  set_property(TARGET moverWallsTest PROPERTY HIP_ARCHITECTURES ${HIP_ARCH})
  target_compile_definitions(moverWallsTest PUBLIC HIPIFLY)
endif()

enable_testing()

foreach(test_case invariants box corner turned-back narrow wall-time no-wall
                  relativistic stopped)
  string(REPLACE "-" "_" test_name ${test_case})
  add_test(NAME moverWalls_${test_name}
           COMMAND $<TARGET_FILE:moverWallsTest> --case ${test_case})
  set_tests_properties(moverWalls_${test_name} PROPERTIES TIMEOUT 120)
endforeach()
