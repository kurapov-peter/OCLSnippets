cmake_minimum_required(VERSION 3.10)
project(OCLSnippets)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(oclhelpers 0.1.0 REQUIRED)

function(add_kernel name)
  configure_file(${name}.cl ${CMAKE_CURRENT_BINARY_DIR}/${name}.cl COPYONLY)
endfunction()

add_subdirectory(common)

include_directories(${PROJECT_SOURCE_DIR})
add_subdirectory(constant)
