cmake_minimum_required(VERSION 3.21)
project(restartOutputFilesTest LANGUAGES CXX)

# The serial-HDF5 output files on a restart (restartOutputFiles): the
# datasets cycle_<N> from the restart cycle on removed in every group, the
# others kept; the topology of a rank's file and the decomposition of
# settings.hdf recognised; the rank files of a folder listed. CPU only, no
# MPI calls; built with USE_HDF5=ON (the HDF5 targets of the main project).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(restartOutputFilesTest
               ${CMAKE_CURRENT_SOURCE_DIR}/restartOutputFilesTest.cpp
               ${PROJECT_ROOT}/src/ipic3d/inputoutput/restartOutputFiles.cpp
               ${PROJECT_ROOT}/src/ipic3d/inputoutput/outputPrepare.cpp)
target_include_directories(restartOutputFilesTest
                           PUBLIC ${PROJECT_ROOT}/src/include)
# MPI for a parallel HDF5, whose hdf5.h includes mpi.h
target_link_libraries(restartOutputFilesTest PRIVATE ${HDF5_C_HL_TARGET}
                                                     ${HDF5_C_TARGET} MPI::MPI_CXX)

enable_testing()
include(${CMAKE_CURRENT_SOURCE_DIR}/../unitTestLaunch.cmake)
add_unit_test(restart_output_files TARGET restartOutputFilesTest TIMEOUT 60)
