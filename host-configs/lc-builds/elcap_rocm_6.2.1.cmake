set(
    HIP_ON
    ON
    CACHE BOOL
    "Enable HIP support"
)
set(
    CMAKE_C_COMPILER
    "/usr/tce/packages/cray-mpich/cray-mpich-8.1.31-rocmcc-6.2.1-magic/bin/mpiamdclang"
    CACHE STRING
    "C compiler for El Cap"
)
set(
    CMAKE_CXX_COMPILER
    "/usr/tce/packages/cray-mpich/cray-mpich-8.1.31-rocmcc-6.2.1-magic/bin/mpiamdclang++"
    CACHE STRING
    "C++ compiler for El Cap"
)
set(
    CMAKE_HIP_COMPILER
    "/usr/tce/packages/cray-mpich/cray-mpich-8.1.31-rocmcc-6.2.1-magic/bin/mpiamdclang++"
    CACHE STRING
    "HIP compiler for El Cap"
)
set(
    HIP_DIR
    "/usr/tce/packages/rocmcc/rocmcc-6.2.1-magic/lib/cmake/hip"
    CACHE STRING
    "Directory where CMake will look for hip-config.cmake"
)
