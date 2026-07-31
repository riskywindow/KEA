# Command line tools. Each tool lives in its own subdirectory and is added only
# if present, so tools can be developed independently without this file
# becoming a merge point.
foreach(_tool kea-as kea-dis kea-sim kea-rt keac)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_tool}/CMakeLists.txt)
    add_subdirectory(${_tool})
  endif()
endforeach()
