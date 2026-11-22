cmake_minimum_required(VERSION 3.21)
project(asyncOutputTest LANGUAGES CXX)

# The deferred output (AsyncOutput): the order of the writes of a lane and of
# the completions, the waits of the copies into the host buffers and of a
# lane's library, a writer's error on the main thread, the synchronous mode.
# CPU only, no MPI.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(asyncOutputTest
               ${CMAKE_CURRENT_SOURCE_DIR}/asyncOutputTest.cpp
               ${PROJECT_ROOT}/src/ipic3d/inputoutput/AsyncOutput.cpp)
target_include_directories(asyncOutputTest PUBLIC ${PROJECT_ROOT}/src/include)
target_link_libraries(asyncOutputTest PRIVATE Threads::Threads)

enable_testing()
add_test(NAME asyncOutput COMMAND asyncOutputTest)
set_tests_properties(asyncOutput PROPERTIES TIMEOUT 60)
