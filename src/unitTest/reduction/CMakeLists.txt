cmake_minimum_required(VERSION 3.21) # HIP requires 3.21
project(reductionTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


option(CUDA_ON "Enable CUDA" ON) # It should always be ON, even for HIP AMD
option(HIP_ON "Use HIP" OFF) # Set it to OFF if you want to use CUDA

if(NOT HIP_ON) # CUDA
    enable_language(CUDA)
    message(STATUS "Using CUDA environment.")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

elseif(HIP_ON AND CUDA_ON) # HIP
    find_package(HIP REQUIRED)
    enable_language(HIP)
    message(STATUS "Using HIP environment.")
    set(DEVICE_LANGUAGE HIP)
    set(DEVICE_COMPILER hipcc)

endif()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(SOURCE_CODE_DIR ${PROJECT_ROOT}/src/ipic3d)
set(INCLUDE_CODE_DIR ${PROJECT_ROOT}/src/include)

file(
  GLOB
  src_files
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  )
# CUDA source files
if(CUDA_ON)
  file(GLOB cuda_src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
  if(HIP_ON)
    set_source_files_properties(${cuda_src_files} PROPERTIES LANGUAGE HIP)
  endif()
  list(APPEND src_files ${cuda_src_files})
endif()

add_executable(reductionTest ${src_files})

set(include_path ${INCLUDE_CODE_DIR})
if(CUDA_ON)
  list(APPEND include_path ${INCLUDE_CODE_DIR}/CUDA)
  list(APPEND include_path ${INCLUDE_CODE_DIR}/dataAnalysis)
  list(APPEND include_path ${INCLUDE_CODE_DIR}/dataAnalysis/GMM)
  list(APPEND include_path ${INCLUDE_CODE_DIR}/dataAnalysis/Histogram)
  if(NOT HIP_ON)
    list(APPEND include_path ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  else()
    list(APPEND include_path ${INCLUDE_CODE_DIR}/hip)
  endif()
endif()
target_include_directories(
  reductionTest
  PUBLIC ${include_path}
  )

if(NOT HIP_ON) # CUDA

  set_target_properties(reductionTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET reductionTest PROPERTY CUDA_ARCHITECTURES 75) # required for double atomicadd
  set_property(TARGET reductionTest PROPERTY CUDA_STANDARD 17)
  set_property(TARGET reductionTest PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET reductionTest PROPERTY LINK_OPTIONS -fno-lto)

elseif(HIP_ON AND CUDA_ON) # HIP

  target_link_libraries(reductionTest PRIVATE hip::device)
  target_link_libraries(reductionTest PRIVATE hip::host)

  target_compile_options(reductionTest PUBLIC -fgpu-rdc)
  target_link_options(reductionTest PUBLIC  -fgpu-rdc --hip-link)

  set_property(TARGET reductionTest PROPERTY HIP_ARCHITECTURES gfx90a)

endif()

if (CUDA_ON)
  target_compile_definitions(reductionTest PUBLIC CUDA_ON=true)

  if(MOMENT_CUDA_ON)
    target_compile_definitions(reductionTest PUBLIC MOMENT_CUDA_ON=true)
  else()
    target_compile_definitions(reductionTest PUBLIC MOMENT_CUDA_ON=false)
  endif()
  if(MOVER_CUDA_ON)
    target_compile_definitions(reductionTest PUBLIC MOVER_CUDA_ON=true)
  else()
    target_compile_definitions(reductionTest PUBLIC MOVER_CUDA_ON=false)
  endif()

  if(HIP_ON)
  target_compile_definitions(reductionTest PUBLIC HIPIFLY)
  endif()
else()
  target_compile_definitions(reductionTest PUBLIC CUDA_ON=false)
  target_compile_definitions(reductionTest PUBLIC MOMENT_CUDA_ON=false)
  target_compile_definitions(reductionTest PUBLIC MOVER_CUDA_ON=false)
endif()


# TEST

include(CTest)
enable_testing()

# Test cases
add_test(NAME reductionTest1 COMMAND reductionTest)
set_tests_properties(reductionTest1 PROPERTIES TIMEOUT 10) # Timeout in seconds