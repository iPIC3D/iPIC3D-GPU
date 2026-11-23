cmake_minimum_required(VERSION 3.21)
project(threadErrorsTest LANGUAGES CXX)

# An error raised in a ThreadPool task or on a secondary thread of one rank
# ends every rank (MPIdata: std::terminate -> MPI_Abort), so that no rank is
# left waiting in a collective. CPU only.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(INCLUDE_CODE_DIR ${PROJECT_ROOT}/src/include)

add_executable(threadErrorsTest
               ${CMAKE_CURRENT_SOURCE_DIR}/threadErrorsTest.cpp
               ${PROJECT_ROOT}/src/ipic3d/utility/MPIdata.cpp)
target_include_directories(threadErrorsTest PUBLIC ${INCLUDE_CODE_DIR})
target_link_libraries(threadErrorsTest PRIVATE MPI::MPI_CXX)

enable_testing()

# each case: the message of rank 1 (std::terminate), then MPI_Abort ending
# the run (runCase.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../unitTestLaunch.cmake)
unit_test_environment(launch_environment)
add_test(NAME thread_errors
         COMMAND ${CMAKE_COMMAND}
                 -DLAUNCH=${CMAKE_CURRENT_SOURCE_DIR}/../launch.sh
                 -DEXE=$<TARGET_FILE:threadErrorsTest>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/runCase.cmake)
set_tests_properties(thread_errors PROPERTIES PROCESSORS 2 TIMEOUT 90
                                              ENVIRONMENT "${launch_environment}")
