cmake_minimum_required(VERSION 3.21)
project(threadErrorsTest LANGUAGES CXX)

# An error raised on a secondary host thread (a ThreadPool task, a
# std::thread, an OpenMP thread) or on the main thread of one rank ends every
# rank (MPIdata: std::terminate -> MPI_Abort), so that no rank is left waiting
# in a collective. CPU only.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP COMPONENTS CXX)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(INCLUDE_CODE_DIR ${PROJECT_ROOT}/src/include)

add_executable(threadErrorsTest
               ${CMAKE_CURRENT_SOURCE_DIR}/threadErrorsTest.cpp
               ${PROJECT_ROOT}/src/ipic3d/utility/MPIdata.cpp)
target_include_directories(threadErrorsTest PUBLIC ${INCLUDE_CODE_DIR})
target_link_libraries(threadErrorsTest PRIVATE MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(threadErrorsTest PRIVATE OpenMP::OpenMP_CXX)
endif()

enable_testing()

# each case: the message of rank 1 (std::terminate), then MPI_Abort ending
# the run (runCase.cmake); the negative control abort (std::abort on rank 1,
# no handler) must fail that check
foreach(test_case task thread openmp main abort)
  set(expect "injected failure")
  if(test_case STREQUAL "task")
    set(expect "test task: injected failure")
  endif()
  add_test(NAME threadErrors_${test_case}
           COMMAND ${CMAKE_COMMAND} -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                   -DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                   "-DPREFLAGS=${MPIEXEC_PREFLAGS}"
                   "-DPOSTFLAGS=${MPIEXEC_POSTFLAGS}"
                   -DEXE=$<TARGET_FILE:threadErrorsTest> -DCASE=${test_case}
                   "-DEXPECT=${expect}"
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/runCase.cmake)
  set_tests_properties(threadErrors_${test_case} PROPERTIES PROCESSORS 2
                                                            TIMEOUT 60)
endforeach()
set_tests_properties(threadErrors_abort PROPERTIES WILL_FAIL TRUE)
