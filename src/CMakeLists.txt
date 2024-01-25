cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(SURFACE_RECONSTRUCTION)
find_package(PCL 1.3 REQUIRED)

# When PCL is found, several related variables are set:

#     PCL_FOUND: set to 1 if PCL is found, otherwise unset

#     PCL_INCLUDE_DIRS: set to the paths to PCL installed headers and the dependency headers

#     PCL_LIBRARIES: set to the file names of the built and installed PCL libraries

#     PCL_LIBRARY_DIRS: set to the paths to where PCL libraries and 3rd party dependencies reside

#     PCL_VERSION: the version of the found PCL

#     PCL_COMPONENTS: lists all available components

#     PCL_DEFINITIONS: lists the needed preprocessor definitions and compiler flags
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_read pcl_read.cpp)
target_link_libraries(pcl_read ${PCL_LIBRARIES})
