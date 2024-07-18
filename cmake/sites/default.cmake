if(NOT "${CMAKE_CURRENT_LIST_FILE}" STREQUAL "${iPic3D_SOURCE_DIR}/cmake/sites/default.cmake")
  message(FATAL_ERROR "Site-specific setup has gone wrong")
endif()

# This being the default, we don't set anything.
