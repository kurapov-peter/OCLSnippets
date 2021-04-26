#define CL_VERSION_1_2
#include <common/common.h>
#include <iostream>
#include <tuple>
#include <vector>

#include <boost/program_options.hpp>

namespace ocl = oclhelpers;

const std::string kernel_filename = "constant.cl";

void run_kernel(const std::string &filename) {
  auto p = ocl::get_platforms()[0];
  auto [platform, device, ctx, program] =
      ocl::compile_file_with_default_gpu(p, filename);
  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl
            << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  constexpr int buffer_size = 1;
  cl::Buffer src(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  cl::Buffer dst(ctx, CL_MEM_READ_WRITE, sizeof(int) * buffer_size);
  std::vector<int> src_data = {11};
  std::vector<int> dst_data = {-1};

  cl_int queue_err;
  cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES,
                                          CL_QUEUE_PROFILING_ENABLE, 0};
  cl::CommandQueue queue(ctx, device, props, &queue_err);
  if (queue_err != CL_SUCCESS) {
    std::cerr << "Queue creation error: ";
    std::cerr << ocl::get_error_string(queue_err) << std::endl;
  }

  //   auto event = std::make_unique<cl::Event>();
  cl::Event event;
  OCL_SAFE_CALL(queue.enqueueWriteBuffer(
      src, CL_TRUE, 0, sizeof(int) * buffer_size, src_data.data()));
  OCL_SAFE_CALL(queue.enqueueWriteBuffer(
      dst, CL_TRUE, 0, sizeof(int) * buffer_size, dst_data.data()));
  cl::Kernel kernel = cl::Kernel(program, "constant_kernel");
  ocl::set_args(kernel, src, dst);

  OCL_SAFE_CALL(queue.enqueueNDRangeKernel(
      kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, {}, &event));
  OCL_SAFE_CALL(queue.enqueueReadBuffer(
      src, CL_TRUE, 0, sizeof(int) * buffer_size, src_data.data()));
  OCL_SAFE_CALL(queue.enqueueReadBuffer(
      dst, CL_TRUE, 0, sizeof(int) * buffer_size, dst_data.data()));
  OCL_SAFE_CALL(queue.finish());
  event.wait();

  auto status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();

  // cl_int status;
  // event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);
  if (status != CL_COMPLETE) {
    std::cout << "Status is " << status << " " << ocl::get_error_string(status)
              << " (should be CL_COMPLETE)";
  }

  cl_int profiling_error1;
  cl_int profiling_error2;
  auto exe_time =
      event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&profiling_error1) -
      event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&profiling_error2);
  // cl_ulong start, end;
  // profiling_error1 = event.getProfilingInfo(CL_PROFILING_COMMAND_START,
  // &start); profiling_error2 =
  // event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
  if (profiling_error1 != CL_SUCCESS || profiling_error2 != CL_SUCCESS) {
    std::cerr << "Got profiling error: ";
    std::cerr << ocl::get_error_string(profiling_error1) << " & ";
    std::cerr << ocl::get_error_string(profiling_error2) << std::endl;
  }

  std::cout << "Execution time: " << exe_time << "ns" << std::endl;

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
