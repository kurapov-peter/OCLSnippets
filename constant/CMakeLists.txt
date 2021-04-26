
add_kernel(constant)
add_executable(constant constant.cpp)
target_link_libraries(constant PRIVATE common Boost::program_options)

add_executable(timed-constant timed_constant.cpp)
target_link_libraries(timed-constant PRIVATE common Boost::program_options)