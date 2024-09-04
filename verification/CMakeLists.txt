cmake_minimum_required(VERSION 3.10)

project(CompareVTKFiles)

# Find VTK
find_package(VTK COMPONENTS 
  CommonCore
  CommonDataModel
  IOCore
  IOLegacy
  REQUIRED
)

# Create executable
add_executable(CompareVTKFiles verification.cpp)

# Link VTK libraries
target_link_libraries(CompareVTKFiles ${VTK_LIBRARIES})
