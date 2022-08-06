#include <torch/extension.h>

void launch_triplane_sampler_forward_kernel(
        torch::Tensor output, torch::Tensor input, torch::Tensor grid,
        int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

//
void launch_triplane_sampler_backward_kernel(
        const torch::Tensor grad_input, const torch::Tensor grad_grid,
        const torch::Tensor grad_output, const torch::Tensor input,
        const torch::Tensor grid, int64_t interpolation_mode, int64_t padding_mode,
        bool align_corners, std::array<bool, 2> output_mask);


torch::Tensor triplane_sampler_forward_cuda(torch::Tensor input, torch::Tensor grid,
                                            int64_t interpolation_mode, int64_t padding_mode,
                                            bool align_corners) {
    auto in_size = input.sizes();
    auto grid_size = grid.sizes();
    auto output = input.new_zeros({in_size[0], in_size[1] / 3, grid_size[1], grid_size[2]});
    launch_triplane_sampler_forward_kernel(
            output, input, grid, interpolation_mode, padding_mode, align_corners);
    return output;
}

std::tuple <torch::Tensor, torch::Tensor>
triplane_sampler_backward_cuda(const torch::Tensor grad_output, const torch::Tensor input,
                               const torch::Tensor grid, int64_t interpolation_mode, int64_t padding_mode,
                               bool align_corners, std::array<bool, 2> output_mask) {
    auto input_requires_grad = output_mask[0];
    auto grid_requires_grad = output_mask[1];

    torch::Tensor grad_input = ([&]() {
        if (input_requires_grad) {
            return torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        } else {
            return input.new_zeros({1, 1, 1, 1});
        }
    })();
    torch::Tensor grad_grid = ([&]() {
        if (grid_requires_grad) {
            return torch::zeros_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        } else {
            return torch::zeros({0, 0, 0, 0});
        }
    })();

    launch_triplane_sampler_backward_kernel(
            grad_input, grad_grid, grad_output, input,
            grid, interpolation_mode, padding_mode, align_corners, output_mask);
    return std::make_tuple(grad_input, grad_grid);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("triplane_sampler_forward", &triplane_sampler_forward_cuda, "triplane sampler forward (CUDA)");
m.def("triplane_sampler_backward", &triplane_sampler_backward_cuda, "triplane sampler backward (CUDA)");
}
