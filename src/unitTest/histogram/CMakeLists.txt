cmake_minimum_required(VERSION 3.21) # HIP requires 3.21
project(histogramTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(HIP_ON "Use HIP" OFF) # Set it to OFF if you want to use CUDA

if(NOT HIP_ON) # CUDA
  enable_language(CUDA)
  message(STATUS "Using CUDA environment.")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

elseif() # HIP
  find_package(HIP REQUIRED)
  enable_language(HIP)
  message(STATUS "Using HIP environment.")
  set(DEVICE_LANGUAGE HIP)
  set(DEVICE_COMPILER hipcc)

endif()

find_package(MPI REQUIRED COMPONENTS CXX)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(SOURCE_CODE_DIR ${PROJECT_ROOT}/src/ipic3d)
set(INCLUDE_CODE_DIR ${PROJECT_ROOT}/src/include)

# source 
file(
  GLOB
  src_files
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
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

list(REMOVE_ITEM src_files ${SOURCE_CODE_DIR}/iPIC3D.cpp)

file(GLOB cuda_src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${SOURCE_CODE_DIR}/cudaKernel/*.cu ${SOURCE_CODE_DIR}/main/*.cu ${SOURCE_CODE_DIR}/dataAnalysis/*.cu)
if(HIP_ON)
  set_source_files_properties(${cuda_src_files} PROPERTIES LANGUAGE HIP)
endif()
list(APPEND src_files ${cuda_src_files})

add_executable(histogramTest ${src_files})

# include
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

target_include_directories(
  histogramTest
  PUBLIC ${include_path}
  )

set(linkLibraries MPI::MPI_CXX)  

target_link_libraries(histogramTest PRIVATE ${linkLibraries})

# configure GPU
if(NOT HIP_ON) # CUDA

  set_target_properties(histogramTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET histogramTest PROPERTY CUDA_ARCHITECTURES 75) # required for double atomicadd
  set_property(TARGET histogramTest PROPERTY CUDA_STANDARD 17)
  set_property(TARGET histogramTest PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET histogramTest PROPERTY LINK_OPTIONS -fno-lto)

else() # HIP

  target_link_libraries(histogramTest PRIVATE hip::device)
  target_link_libraries(histogramTest PRIVATE hip::host)

  target_compile_options(histogramTest PUBLIC -fgpu-rdc)
  target_link_options(histogramTest PUBLIC  -fgpu-rdc --hip-link)

  set_property(TARGET histogramTest PROPERTY HIP_ARCHITECTURES gfx90a)

endif()


if(HIP_ON)
target_compile_definitions(histogramTest PUBLIC HIPIFLY)
endif()

target_compile_definitions(histogramTest PUBLIC NO_HDF5)


# TEST

include(CTest)
enable_testing()

# Test cases
add_test(NAME histogramTest1 COMMAND histogramTest)
set_tests_properties(histogramTest1 PROPERTIES TIMEOUT 10) # Timeout in seconds