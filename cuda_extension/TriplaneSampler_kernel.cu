#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/GridSampler.cuh>

using namespace at::cuda::detail;
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
using namespace at::native;

template<typename scalar_t, typename index_t>
__global__ void triplane_sampler_forward_kernel(
        const index_t nthreads,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
        const GridSamplerInterpolation interpolation_mode,
        const GridSamplerPadding padding_mode,
        bool align_corners) {
    index_t C = input.size(1) / 3;
    index_t inp_H = input.size(2);
    index_t inp_W = input.size(3);
    index_t out_H = grid.size(1);
    index_t out_W = grid.size(2);

//    index_t index = blockIdx.x * blockDim.x + threadIdx.x;
//    if (index < nthreads)
    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t)
    {
        const index_t w = index % out_W;
        const index_t h = (index / out_W) % out_H;
        const index_t n = index / (out_H * out_W);

        for (int plane_idx = 0; plane_idx < 3; plane_idx++) {
            scalar_t x = grid[n][h][w][plane_idx];
            scalar_t y = grid[n][h][w][(plane_idx + 1) % 3];

            scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
            scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

            if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get NE, NW, SE, SW pixel values from (x, y)
                index_t ix_nw = static_cast<index_t>(::floor(ix));
                index_t iy_nw = static_cast<index_t>(::floor(iy));
                index_t ix_ne = ix_nw + 1;
                index_t iy_ne = iy_nw;
                index_t ix_sw = ix_nw;
                index_t iy_sw = iy_nw + 1;
                index_t ix_se = ix_nw + 1;
                index_t iy_se = iy_nw + 1;

                // get surfaces to each neighbor:
                scalar_t nw = (ix_se - ix) * (iy_se - iy);
                scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
                scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
                scalar_t se = (ix - ix_nw) * (iy - iy_nw);

                // calculate bilinear weighted pixel value and set output pixel
                for (index_t c = 0; c < C; ++c) {
                    index_t triplane_c = c + C * plane_idx;
                    if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                        output[n][c][h][w] += input[n][triplane_c][iy_nw][ix_nw] * nw;
                    }
                    if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                        output[n][c][h][w] += input[n][triplane_c][iy_ne][ix_ne] * ne;
                    }
                    if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                        output[n][c][h][w] += input[n][triplane_c][iy_sw][ix_sw] * sw;
                    }
                    if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                        output[n][c][h][w] += input[n][triplane_c][iy_se][ix_se] * se;
                    }
                }
            } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                index_t ix_nearest = static_cast<index_t>(::round(ix));
                index_t iy_nearest = static_cast<index_t>(::round(iy));

                // assign nearest neighor pixel value to output pixel
                for (index_t c = 0; c < C; ++c) {
                    index_t triplane_c = c + C * plane_idx;
                    if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                        output[n][c][h][w] = input[n][triplane_c][iy_nearest][ix_nearest];
                    } else {
                        output[n][c][h][w] = static_cast<scalar_t>(0);
                    }
                }
            }
        }
    }
}

template<typename scalar_t, typename index_t>
__global__ void triplane_sampler_backward_kernel(
        const index_t nthreads,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_output,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_grid,   // initialized to empty
        const GridSamplerInterpolation interpolation_mode,
        const GridSamplerPadding padding_mode,
        bool align_corners,
        const bool input_requires_grad,
        const bool grid_requires_grad,
        const index_t grad_input_memory_span) {
    index_t C = input.size(1) / 3;
    index_t inp_H = input.size(2);
    index_t inp_W = input.size(3);
    index_t out_H = grid.size(1);
    index_t out_W = grid.size(2);

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t)
    {
        const index_t w = index % out_W;
        const index_t h = (index / out_W) % out_H;
        const index_t n = index / (out_H * out_W);

        for (int plane_idx = 0; plane_idx < 3; plane_idx++) {
            // get the corresponding input x, y co-ordinates from grid
            scalar_t x = grid[n][h][w][plane_idx];
            scalar_t y = grid[n][h][w][(plane_idx + 1) % 3];

            // multipliers for gradients on ix and iy
            scalar_t gix_mult, giy_mult;
            scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners,
                                                                     &gix_mult);
            scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners,
                                                                     &giy_mult);

            if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                // get NE, NW, SE, SW pixel values from (x, y)
                index_t ix_nw = static_cast<index_t>(::floor(ix));
                index_t iy_nw = static_cast<index_t>(::floor(iy));
                index_t ix_ne = ix_nw + 1;
                index_t iy_ne = iy_nw;
                index_t ix_sw = ix_nw;
                index_t iy_sw = iy_nw + 1;
                index_t ix_se = ix_nw + 1;
                index_t iy_se = iy_nw + 1;

                // get surfaces to each neighbor:
                scalar_t nw = (ix_se - ix) * (iy_se - iy);
                scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
                scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
                scalar_t se = (ix - ix_nw) * (iy - iy_nw);

                scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
                for (index_t c = 0; c < C; ++c) {
                    index_t triplane_idx = c + C * plane_idx;
                    scalar_t gOut = grad_output[n][c][h][w];

                    if (input_requires_grad) {
                        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
                        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                            gpuAtomicAdd(&grad_input[n][triplane_idx][iy_nw][ix_nw], nw * gOut);
                        }
                        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                            gpuAtomicAdd(&grad_input[n][triplane_idx][iy_ne][ix_ne], ne * gOut);
                        }
                        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                            gpuAtomicAdd(&grad_input[n][triplane_idx][iy_sw][ix_sw], sw * gOut);
                        }
                        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                            gpuAtomicAdd(&grad_input[n][triplane_idx][iy_se][ix_se], se * gOut);
                        }
                    }

                    if (grid_requires_grad) {
                        // calculate grad_grid
                        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                            scalar_t nw_val = input[n][triplane_idx][iy_nw][ix_nw];
                            gix -= nw_val * (iy_se - iy) * gOut;
                            giy -= nw_val * (ix_se - ix) * gOut;
                        }
                        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                            scalar_t ne_val = input[n][triplane_idx][iy_ne][ix_ne];
                            gix += ne_val * (iy_sw - iy) * gOut;
                            giy -= ne_val * (ix - ix_sw) * gOut;
                        }
                        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                            scalar_t sw_val = input[n][triplane_idx][iy_sw][ix_sw];
                            gix -= sw_val * (iy - iy_ne) * gOut;
                            giy += sw_val * (ix_ne - ix) * gOut;
                        }
                        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                            scalar_t se_val = input[n][triplane_idx][iy_se][ix_se];
                            gix += se_val * (iy - iy_nw) * gOut;
                            giy += se_val * (ix - ix_nw) * gOut;
                        }
                    }
                }

                // assuming grad_grid is contiguous
                // thus we can
                //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
                //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
                if (grid_requires_grad) {
                    grad_grid[n][h][w][plane_idx] += gix_mult * gix;
                    grad_grid[n][h][w][(plane_idx + 1) % 3] += giy_mult * giy;
                }
            } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                if (input_requires_grad) {
                    index_t ix_nearest = static_cast<index_t>(::round(ix));
                    index_t iy_nearest = static_cast<index_t>(::round(iy));

                    // assign nearest neighor pixel value to output pixel
                    for (index_t c = 0; c < C; ++c) {
                        index_t triplane_idx = c + C * plane_idx;
                        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
                        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                            fastAtomicAdd(&grad_input[n][triplane_idx][iy_nearest][ix_nearest], 0,
                                          grad_input_memory_span, grad_output[n][c][h][w], true);
//                            gpuAtomicAdd(&grad_input[n][triplane_idx][iy_nearest][ix_nearest], grad_output[n][c][h][w]);
                        }
                    }
                }

                // assuming grad_grid is contiguous
                // thus we can
                //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
                //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
//                grad_grid[n][h][w][plane_idx] = static_cast<scalar_t>(0);
//                grad_grid[n][h][w][(plane_idx + 1) % 3] = static_cast<scalar_t>(0);
            }
        }
    }
}

void launch_triplane_sampler_forward_kernel(
        torch::Tensor output, torch::Tensor input, torch::Tensor grid,
        int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    // See NOTE [ grid_sampler Native Functions ].
    // Add checks here in case this is called instead of grid_sampler.
//    check_grid_sampler_common(input, grid);
//    check_grid_sampler_2d(input, grid);

    auto N = input.size(0);
    auto H = grid.size(1);
    auto W = grid.size(2);
    int64_t count = N * H * W;
    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "triplane_sampler_forward", [&] {
            triplane_sampler_forward_kernel<scalar_t>
            <<<GET_BLOCKS(count, 1024), 1024>>>(
                    count,
                    input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                    static_cast<GridSamplerPadding>(padding_mode),
                    align_corners);
        });
    }
}

//
void launch_triplane_sampler_backward_kernel(
        torch::Tensor grad_input, const torch::Tensor grad_grid,
        const torch::Tensor grad_output, const torch::Tensor input,
        const torch::Tensor grid, int64_t interpolation_mode, int64_t padding_mode,
        bool align_corners, std::array<bool, 2> output_mask) {
    // See NOTE [ grid_sampler Native Functions ].
    // Add checks here in case this is called instead of grid_sampler.
//    check_grid_sampler_common(input, grid);
//    check_grid_sampler_2d(input, grid);

    // See Note [Writing Nondeterministic Operations]
    // Nondeterministic because of atomicAdd usage
//    globalContext().alertNotDeterministic("my_grid_sampler_2d_backward_cuda");
    auto N = input.size(0);
    auto H = grid.size(1);
    auto W = grid.size(2);

    // If `input` gradient is not required, we skip computing it -- not needing to create
    // the tensor to hold the gradient can markedly increase performance. (`grid` gradient
    // is always computed.)
    auto input_requires_grad = output_mask[0];
    auto grid_requires_grad = output_mask[1];

    int64_t count = N * H * W;
    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "triplane_sampler_backward", [&] {
            triplane_sampler_backward_kernel<scalar_t>
            <<<GET_BLOCKS(count, 1024), 1024>>>(
                    static_cast<int>(count),
                    grad_output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    grad_input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    grad_grid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                    static_cast<GridSamplerPadding>(padding_mode),
                    align_corners,
                    input_requires_grad,
                    grid_requires_grad,
                    static_cast<int>(grad_input.numel()));
        });
    }
}
