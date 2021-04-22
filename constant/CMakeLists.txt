
add_kernel(constant)
add_executable(constant constant.cpp)
target_link_libraries(constant PRIVATE common)
