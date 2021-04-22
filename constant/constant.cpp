#include <common/common.h>
#include <iostream>
#include <tuple>
#include <vector>

namespace ocl = oclhelpers;

const std::string kernel_filename = "constant.cl";

int main() {
  auto [platform, device, ctx, program] =
      ocl::compile_file_with_defaults(kernel_filename);
  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl
            << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  constexpr int buffer_size = 1;
  cl::Buffer src(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  cl::Buffer dst(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  std::vector<int> src_data = {11};
  std::vector<int> dst_data = {-1};

  cl::CommandQueue queue(ctx, device);
  queue.enqueueWriteBuffer(src, CL_TRUE, 0, sizeof(int) * buffer_size,
                           src_data.data());
  queue.enqueueWriteBuffer(dst, CL_TRUE, 0, sizeof(int) * buffer_size,
                           dst_data.data());
  cl::Kernel kernel = cl::Kernel(program, "constant_kernel");
  //   ocl::set_args(kernel, src, dst);
  kernel.setArg(0, src);
  kernel.setArg(1, dst);

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1),
                             cl::NullRange);
  queue.enqueueReadBuffer(src, CL_TRUE, 0, sizeof(int) * buffer_size,
                          src_data.data());
  queue.enqueueReadBuffer(dst, CL_TRUE, 0, sizeof(int) * buffer_size,
                          dst_data.data());
  queue.finish();

  std::cout << "Got result: " << dst_data[0] << "\n"
            << "Expected: 42\n"
            << "Src is: " << src_data[0] << "\n";
  return 0;
}
