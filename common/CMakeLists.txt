add_library(common common.h)
target_link_libraries(common PUBLIC OpenCL::OpenCL oclhelpers::oclhelpers)
