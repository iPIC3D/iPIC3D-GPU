cmake_minimum_required(VERSION 3.21)
project(analysisRecordsTest LANGUAGES CXX)

# The files of the data-analysis records on a restart (outputPrepare): the
# binary records of RecordFile cut from the restart cycle on, also after a
# record cut short, and the files kept aside; the GMM records and the index
# of sparse bins. CPU only.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(analysisRecordsTest
               ${CMAKE_CURRENT_SOURCE_DIR}/analysisRecordsTest.cpp
               ${PROJECT_ROOT}/src/ipic3d/inputoutput/outputPrepare.cpp
               ${PROJECT_ROOT}/src/ipic3d/inputoutput/gmmRecords.cpp
               ${PROJECT_ROOT}/src/ipic3d/inputoutput/sparseIndex.cpp)
target_include_directories(analysisRecordsTest
                           PUBLIC ${PROJECT_ROOT}/src/include)

enable_testing()
add_test(NAME analysisRecords COMMAND analysisRecordsTest)
set_tests_properties(analysisRecords PROPERTIES TIMEOUT 60)
