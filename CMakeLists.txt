cmake_minimum_required(VERSION 3.21)

project(iPic3D LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

macro(DP var)
  message(NOTICE "${var} = '${${var}}'")
endmacro()

# First, do the site configuration

# This function globs for the existing configurations.
function(get_sites OUTVAR)
  file(GLOB site_incs RELATIVE "${CMAKE_SOURCE_DIR}/cmake/sites" "${CMAKE_SOURCE_DIR}/cmake/sites/*.cmake")
  set(sites "")
  foreach(inc "${site_incs}")
    string(REPLACE ".cmake" "" site "${inc}")
    list(APPEND sites "${site}")
  endforeach()
  list(FIND sites "default" def_ind)
  list(REMOVE_ITEM sites "default")
  list(PREPEND sites "default")
  set(${OUTVAR} "${sites}" PARENT_SCOPE)
endfunction()
get_sites(KNOWN_SITES)

# Set up cache var for which we are using
set(SITE "default" CACHE STRING "A predefined site configuration to use")
# Set the options for the UI
set_property(CACHE SITE PROPERTY STRINGS "${KNOWN_SITES}")
# Include it
include("${CMAKE_SOURCE_DIR}/cmake/sites/${SITE}.cmake")

# Use CMake default variable for this so add_library will
# automatically do the right thing.
option(BUILD_SHARED_LIBS "Use shared libraries if ON, static if OFF" ON)
option(USE_CATALYST "Use Catalyst adaptor" OFF)
option(USE_BATSRUS "Use BATSRUS flag" OFF)
option(USE_HDF5 "Use HDF5 library" ON)
option(USE_PHDF5 "Enable parallel HDF5 output (requires HDF5 with MPI support)" ON)
option(USE_ADIOS2 "Use ADIOS2 library" ON)

if(NOT USE_HDF5)
  set(USE_PHDF5 OFF CACHE BOOL "Disabled because USE_HDF5=OFF" FORCE)
endif()

option(HIP_ON "Use HIP" OFF) # Set it to OFF if you want to use CUDA, ON for HIP

set(CUDA_ARCH "75" CACHE STRING "CUDA compute capability to compile for (minimum 75 for double atomicAdd)")
set(HIP_ARCH "gfx90a" CACHE STRING "HIP GPU architecture to compile for (e.g. gfx90a, gfx940, gfx1100)")

option(BENCH_MARK "Print tasks time" OFF)
option(USE_OPENMP "Use OpenMP in supported loops" ON) # must delete all cmake cache if you change this option
option(GPU_SOLVER "Run the entire field solver on GPU (requires CUDA-aware MPI)" OFF)
option(HALO_OVERLAP "Enable halo-computation overlap in the GPU field solver (requires GPU_SOLVER)" OFF)

include(GNUInstallDirs)

# set directories
set(SOURCE_CODE_DIR src/ipic3d)
set(INCLUDE_CODE_DIR src/include)


# Find ParaView
if(USE_CATALYST)
  find_package(ParaView 5.7 REQUIRED)

  if (NOT TARGET ParaView::PythonCatalyst)
    message(STATUS
      "${CMAKE_PROJECT_NAME} requires ParaView to be built with Catalyst and "
      "Python support enabled. Please rebuild ParaView (or point to a "
      "different build of ParaView) with PARAVIEW_ENABLE_CATALYST and "
      "PARAVIEW_ENABLE_PYTHON set to TRUE")
  else()
    add_library(iPICAdaptor ${CMAKE_SOURCE_DIR}/${SOURCE_CODE_DIR}/catalyst/Adaptor.cxx)
    if (BUILD_SHARED_LIBS)
      install(TARGETS iPICAdaptor)
    endif()

    target_link_libraries(iPICAdaptor PRIVATE ParaView::PythonCatalyst VTK::CommonDataModel Python3::Python)
    target_include_directories(iPICAdaptor INTERFACE ${CMAKE_SOURCE_DIR}/${SOURCE_CODE_DIR}/catalyst)
    target_compile_definitions(iPICAdaptor INTERFACE USE_CATALYST)

  endif()
endif()

# Find HDF5
if (USE_HDF5)
  if(BUILD_SHARED_LIBS)
    set(HDF5_USE_STATIC_LIBRARIES OFF)
  else()
    set(HDF5_USE_STATIC_LIBRARIES ON)
  endif()
  find_package(HDF5 COMPONENTS C HL REQUIRED)


  if (NOT DEFINED HDF5_C_TARGET)
    message(STATUS "Creating a proper target for HDF5")
    #add_library(hdf5::hdf5 INTERFACE IMPORTED)
    set_target_properties(hdf5::hdf5 PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "${HDF5_C_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}"
      )
    set(HDF5_C_TARGET hdf5::hdf5)
  endif()

  if (NOT DEFINED HDF5_C_HL_TARGET)
    message(STATUS "Creating a proper target for HDF5 HL")
    #add_library(hdf5::hdf5_hl INTERFACE IMPORTED)
    set_target_properties(hdf5::hdf5_hl PROPERTIES
      INTERFACE_LINK_LIBRARIES "hdf5::hdf5;${HDF5_C_HL_LIBRARIES}"
      )
    set(HDF5_C_HL_TARGET hdf5::hdf5_hl)
  endif()
endif()


# Find MPI and OpenMP
find_package(MPI REQUIRED COMPONENTS CXX)
if(USE_OPENMP)
find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if (USE_ADIOS2)
  set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/thirdparty/adios2")
  find_package(ADIOS2 REQUIRED)
endif()

#
# Source file list
#
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
  mpidata/*.cpp
  processtopology/*.cpp
  )
# CUDA source files
file(GLOB cuda_src_files ${SOURCE_CODE_DIR}/cudaKernel/*.cu ${SOURCE_CODE_DIR}/main/*.cu ${SOURCE_CODE_DIR}/dataAnalysis/*.cu)

if(HIP_ON)
  set_source_files_properties(${cuda_src_files} PROPERTIES LANGUAGE HIP)
endif()

list(APPEND src_files ${cuda_src_files})

add_library( iPIC3Dlib ${src_files} )


if(HIP_ON)
  find_package(HIP REQUIRED)
  enable_language(HIP)
  message(STATUS "Using HIP environment.")
  set(DEVICE_LANGUAGE HIP)
  set(DEVICE_COMPILER hipcc)
  target_link_libraries(iPIC3Dlib PRIVATE hip::device)
  target_link_libraries(iPIC3Dlib PRIVATE hip::host)

  # On Cray systems the MPI wrapper (CC) handles MPI includes implicitly,
  # but HIP files are compiled with amdclang++ which needs explicit paths.
  if(NOT MPI_CXX_INCLUDE_DIRS AND DEFINED ENV{CRAY_MPICH_DIR})
    set(MPI_CXX_INCLUDE_DIRS "$ENV{CRAY_MPICH_DIR}/include")
    message(STATUS "HIP: Using Cray MPICH include dir: ${MPI_CXX_INCLUDE_DIRS}")
  endif()
  if(MPI_CXX_INCLUDE_DIRS)
    target_include_directories(iPIC3Dlib PUBLIC ${MPI_CXX_INCLUDE_DIRS})
  endif()

  target_compile_options(iPIC3Dlib PUBLIC -fgpu-rdc)
  # PRIVATE: HIP RDC linking is resolved inside the shared library;
  # leaking --hip-link to the CXX executable causes the Cray wrapper
  # to inject GPU offload registration code we don't want.
  target_link_options(iPIC3Dlib PRIVATE -fgpu-rdc --hip-link)

  set_property(TARGET iPIC3Dlib PROPERTY HIP_ARCHITECTURES ${HIP_ARCH})
else()
  enable_language(CUDA)

  if(CUDA_ARCH LESS 75)
    message(FATAL_ERROR "CUDA_ARCH=${CUDA_ARCH} is too low. Minimum 75 is required for double-precision atomicAdd.")
  endif()

  set_target_properties(iPIC3Dlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET iPIC3Dlib PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
  set_property(TARGET iPIC3Dlib PROPERTY CUDA_STANDARD 17)
  set_property(TARGET iPIC3Dlib PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET iPIC3Dlib PROPERTY LINK_OPTIONS -fno-lto)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

set(include_path ${INCLUDE_CODE_DIR})
# CUDA header files
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
  iPIC3Dlib
  PUBLIC ${include_path}
  )

#
# Link external libraries
#
set(linkLibraries MPI::MPI_CXX ${HDF5_C_HL_TARGET})

if(USE_OPENMP)
    list(APPEND linkLibraries OpenMP::OpenMP_CXX)
endif()

if (USE_CATALYST)
    list(APPEND linkLibraries iPICAdaptor)
endif()

if (USE_ADIOS2)
    list(APPEND linkLibraries adios2::cxx_mpi)
endif()

target_link_libraries(
    iPIC3Dlib
    PUBLIC ${linkLibraries}
)



#
# Macro definitions
#
if(USE_BATSRUS)
  target_compile_definitions(iPIC3Dlib PUBLIC BATSRUS)
  message(" WARNING: BATSRUS flag is active.")
else()
  message(" INFO: BATSRUS is not active.")
endif()

if(NOT USE_HDF5)
  target_compile_definitions(iPIC3Dlib PUBLIC NO_HDF5)
endif()

if(USE_PHDF5)
  if(NOT USE_HDF5)
    message(FATAL_ERROR "USE_PHDF5 requires USE_HDF5=ON")
  endif()
  target_compile_definitions(iPIC3Dlib PUBLIC PHDF5)
  message(STATUS "Parallel HDF5 output enabled")
endif()

if (BUILD_SHARED_LIBS)
  install(TARGETS iPIC3Dlib)
endif()


if(HIP_ON)
target_compile_definitions(iPIC3Dlib PUBLIC HIPIFLY)
endif()

if(USE_ADIOS2)
  target_compile_definitions(iPIC3Dlib PUBLIC USE_ADIOS2)
endif()

if(GPU_SOLVER)
  target_compile_definitions(iPIC3Dlib PUBLIC GPU_SOLVER)
  message(STATUS "GPU field solver enabled – requires CUDA-aware MPI at runtime")
endif()

if(HALO_OVERLAP)
  target_compile_definitions(iPIC3Dlib PUBLIC HALO_OVERLAP)
  message(STATUS "Halo-computation overlap enabled for GPU field solver")
endif()


#
# Executable declaration
#

# Particle solver
add_executable(
  iPIC3D
  ${SOURCE_CODE_DIR}/iPIC3D.cpp
  )

target_link_libraries(
  iPIC3D
  iPIC3Dlib
  )


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(DEBUG_MODE)
endif()

if(BENCH_MARK)
target_compile_definitions(iPIC3Dlib PUBLIC LOG_TASKS_TOTAL_TIME)
target_compile_definitions(iPIC3D PUBLIC LOG_TASKS_TOTAL_TIME)
endif()

if(BENCH_MARK)
target_compile_definitions(iPIC3Dlib PUBLIC LOG_TASKS_TOTAL_TIME)
target_compile_definitions(iPIC3D PUBLIC LOG_TASKS_TOTAL_TIME)
endif()

## to save the executable in the folder where the CMakeLists.txt file is, i.e. CMAKE_CURRENT_SOURCE_DIR
set_target_properties(iPIC3D PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

## debug releases have a _d appended to the executable
set_target_properties(iPIC3D PROPERTIES DEBUG_POSTFIX "_d")

# Here we do the RPATH stuff so the exe is relocatable.
# Make it a function to avoid variables leaking.
function(add_install_libdir_to_rpath tgt)
  if(BUILD_SHARED_LIBS)
    get_target_property(install_rpaths ${tgt} INSTALL_RPATH)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND install_rpaths "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      list(APPEND install_rpaths "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    list(REMOVE_DUPLICATES install_rpaths)
    set_target_properties(${tgt}
      PROPERTIES INSTALL_RPATH "${install_rpaths}"
      )
  endif()
endfunction()
add_install_libdir_to_rpath(iPIC3D)

install(TARGETS iPIC3D)

add_subdirectory(share/inputfiles)

message(STATUS "")
message(STATUS "========== iPIC3D Configuration ==========")
message("Which system am I compiling for:")
message("MYHOSTNAME is ${myhostname}")
message(STATUS "  Site config                 : ${SITE}  (cmake/sites/${SITE}.cmake)")
message(STATUS "  Build type                  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libraries            : ${BUILD_SHARED_LIBS}  (ON=shared, OFF=static)")
message(STATUS "  System processor            : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler & compiler flags   :")
message(STATUS "  CMAKE_CXX_COMPILER          : ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_CXX_FLAGS             : ${CMAKE_CXX_FLAGS}")
message(STATUS "  MPI_LIBRARIES               : ${MPI_LIBRARIES}")
if(HIP_ON)
message(STATUS "  GPU backend                 : HIP  (arch=${HIP_ARCH})")
else()
  message(STATUS "  GPU backend                 : CUDA  (arch=sm_${CUDA_ARCH})")
endif()
message(STATUS "  GPU solver                  : ${GPU_SOLVER}  (entire field solver on GPU, requires CUDA-aware MPI at runtime)")
message(STATUS "  GPU halo overlap            : ${HALO_OVERLAP}  (overlap halo communication with computation in GPU field solver, requires GPU_SOLVER=ON)")
message(STATUS "  HDF5 output                 : ${USE_HDF5}")
message(STATUS "  HDF5_INCLUDE_DIRS           : ${HDF5_INCLUDE_DIRS}")
message(STATUS "  HDF5_LIBRARIES              : ${HDF5_LIBRARIES}")
message(STATUS "  HDF5_HL_LIBRARIES           : ${HDF5_HL_LIBRARIES}")
message(STATUS "  Parallel HDF5               : ${USE_PHDF5}  (requires HDF5+MPI; disabled if USE_HDF5=OFF)")
message(STATUS "  ADIOS2 output               : ${USE_ADIOS2} (disabled if USE_ADIOS2=OFF)")
message(STATUS "  OpenMP                      : ${USE_OPENMP} ")
message(STATUS "  Catalyst (in-situ)          : ${USE_CATALYST}")
message(STATUS "  BATSRUS coupling            : ${USE_BATSRUS}")
message(STATUS "  Benchmark timers            : ${BENCH_MARK}  (LOG_TASKS_TOTAL_TIME macro)")
message(STATUS "==========================================")
message(STATUS "")