cmake_minimum_required(VERSION 3.21)
project(solverRunsTest LANGUAGES CXX)

# The whole solver: short runs of the iPIC3D executable on the shipped decks.
# Registered from the main project, which builds that executable.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/../unitTestLaunch.cmake)

add_executable(solverRunsTest ${CMAKE_CURRENT_SOURCE_DIR}/solverRunsTest.cpp)
add_dependencies(solverRunsTest iPIC3D)

enable_testing()
# The test itself only starts the runs and reads their files: each run goes
# through launch.sh, a job step inside a Slurm allocation
unit_test_environment(launch_environment)
add_test(NAME solver_runs
         COMMAND solverRunsTest $<TARGET_FILE:iPIC3D>
                 ${CMAKE_CURRENT_SOURCE_DIR}/../launch.sh
                 ${CMAKE_CURRENT_SOURCE_DIR}/../../../share/inputfiles
                 $<BOOL:${USE_HDF5}> $<BOOL:${USE_PHDF5}> $<BOOL:${USE_ADIOS2}>)
set_tests_properties(solver_runs PROPERTIES TIMEOUT 900
                                            PROCESSORS 4
                                            RESOURCE_LOCK ipic3d_gpu
                                            ENVIRONMENT "${launch_environment}")
