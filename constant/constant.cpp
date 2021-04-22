#include <common/common.h>
#include <iostream>
#include <tuple>
#include <vector>

#include <boost/program_options.hpp>

namespace ocl = oclhelpers;

const std::string kernel_filename = "constant.cl";

void run_kernel(const std::string &filename) {
  auto [platform, device, ctx, program] =
      ocl::compile_file_with_defaults(filename);
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
  ocl::set_args(kernel, src, dst);

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
}

int main(int argc, char *argv[]) {
  namespace po = boost::program_options;
  std::string kernel_filepath;
  po::options_description desc("Constant runner");
  desc.add_options()("help", "Show help message");
  desc.add_options()("kernel_path", po::value<std::string>(&kernel_filepath),
                     "Dwarf to run. List all with 'list' option.");
  po::positional_options_description pos_opts;
  pos_opts.add("kernel_path", 1);

  try {
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .positional(pos_opts)
                  .run(),
              vm);
    vm.notify();

    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }

    if (kernel_filepath.empty()) {
      std::cerr << "Please provide path to kernel." << std::endl;
      return 1;
    }

    run_kernel(kernel_filepath);
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}
