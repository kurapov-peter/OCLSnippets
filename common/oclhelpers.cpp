#include "oclhelpers.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>

namespace oclhelpers {
cl::Platform get_default_platform() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (!platforms.size()) {
    throw OCLHelpersException("No platform found!");
  }
  return platforms[0];
}

std::vector<cl::Device> get_gpus(const cl::Platform &p) {
  std::vector<cl::Device> devices;
  p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (!devices.size()) {
    throw OCLHelpersException("No gpus found!");
  }
  return devices;
}

cl::Device get_default_device(const cl::Platform &p) { return get_gpus(p)[0]; }

std::string read_kernel_from_file(const std::string &filename) {
  std::ifstream is(filename);
  return std::string(std::istreambuf_iterator<char>(is),
                     std::istreambuf_iterator<char>());
}

cl::Program make_program_from_file(cl::Context &ctx,
                                   const std::string &filename) {
  cl::Program::Sources sources;
  auto kernel_code = read_kernel_from_file(filename);
  if (kernel_code.empty()) {
    throw OCLHelpersException("Kernel code is empty!");
  }
  return {ctx, kernel_code};
}

void build(cl::Program &program, cl::Device &device) {
  if (program.build({device}) != CL_SUCCESS) {
    std::stringstream ss;
    ss << "Building failed: "
       << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    throw OCLHelpersException(ss.str());
  }
}

std::tuple<cl::Platform, cl::Device, cl::Context, cl::Program>
compile_file_with_defaults(const std::string &filename) {
  auto platform = get_default_platform();
  auto device = get_default_device(platform);

  cl::Context ctx({device});
  auto program = make_program_from_file(ctx, filename);
  build(program, device);
  return std::make_tuple(platform, device, ctx, program);
}

} // namespace oclhelpers