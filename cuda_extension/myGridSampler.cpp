//#include <ATen/native/cuda/GridSampler.h>
//#include <ATen/Functions.h>
//#include <ATen/NativeFunctions.h>
//
//#include <ATen/ops/empty.h>
//#include <ATen/ops/empty_like.h>
//#include <ATen/ops/grid_sampler_2d_backward_native.h>
//#include <ATen/ops/grid_sampler_2d_native.h>
//#include <ATen/ops/grid_sampler_3d_backward_native.h>
//#include <ATen/ops/grid_sampler_3d_native.h>
//#include <ATen/ops/zeros_like.h>
#include <torch/extension.h>

void my_launch_grid_sampler_2d_forward_kernel(
        torch::Tensor output, torch::Tensor input, torch::Tensor grid,
        int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

//
void my_launch_grid_sampler_2d_backward_kernel(
        const torch::Tensor grad_input, const torch::Tensor grad_grid,
        const torch::Tensor grad_output, const torch::Tensor input,
        const torch::Tensor grid, int64_t interpolation_mode, int64_t padding_mode,
        bool align_corners, std::array<bool, 2> output_mask);


torch::Tensor my_grid_sampler_2d_forward_cuda(torch::Tensor input, torch::Tensor grid,
                                      int64_t interpolation_mode, int64_t padding_mode,
                                      bool align_corners) {
    auto in_size = input.sizes();
    auto grid_size = grid.sizes();
    auto output = input.new_zeros({in_size[0], in_size[1], grid_size[1], grid_size[2]});
    my_launch_grid_sampler_2d_forward_kernel(
            output, input, grid, interpolation_mode, padding_mode, align_corners);
    return output;
}

std::tuple <torch::Tensor, torch::Tensor>
my_grid_sampler_2d_backward_cuda(const torch::Tensor grad_output, const torch::Tensor input,
                                 const torch::Tensor grid, int64_t interpolation_mode, int64_t padding_mode,
                                 bool align_corners, std::array<bool, 2> output_mask) {
    auto input_requires_grad = output_mask[0];
    torch::Tensor grad_input = ([&]() {
        if (input_requires_grad) {
            return torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        } else {
            return torch::Tensor();
        }
    })();
    auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    my_launch_grid_sampler_2d_backward_kernel(
            grad_input, grad_grid, grad_output, input,
            grid, interpolation_mode, padding_mode, align_corners, output_mask);
    return std::make_tuple(grad_input, grad_grid);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("forward", &my_grid_sampler_2d_forward_cuda, "LLTM forward (CUDA)");
m.def("backward", &my_grid_sampler_2d_backward_cuda, "LLTM backward (CUDA)");
}
