cmake_minimum_required(VERSION 2.8.8)
project (ipic_tests)
find_package(MPI REQUIRED)
include_directories(
    ${MPI_INCLUDE_PATH}
)
add_executable(
  basic
  basic.c)
target_link_libraries(
    basic
    ${MPI_LIBRARIES}
)

