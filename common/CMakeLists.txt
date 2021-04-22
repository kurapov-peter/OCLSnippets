add_library(common oclhelpers.cpp)
target_link_libraries(common PUBLIC OpenCL::OpenCL)
