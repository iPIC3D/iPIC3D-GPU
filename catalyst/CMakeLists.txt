#cmake_minimum_required(VERSION 3.3)
#project(simulatorAdaptor VERSION 1.0)

find_package(ParaView 4.1 REQUIRED COMPONENTS vtkPVPythonCatalyst)

include("${PARAVIEW_USE_FILE}")
add_library(catalystAdaptor SHARED Adaptor.cxx)

target_link_libraries(catalystAdaptor vtkPVPythonCatalyst)

