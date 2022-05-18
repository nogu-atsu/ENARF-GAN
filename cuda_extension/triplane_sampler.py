import time
from typing import Tuple

import torch
from torch import nn
from tqdm import tqdm

print("our module")
import triplane_sampler_cuda

GRID_SAMPLE_INTERPOLATION_MODES = {
    "bilinear": 0,
    "nearest": 1,
    "bicubic": 2,
}

GRID_SAMPLE_PADDING_MODES = {
    "zeros": 0,
    "border": 1,
    "reflection": 2,
}


class GridSamplerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='zeros', align_corners=False, output_mask=(True, True)):
        """
        grid_sample feature and probability for each part from triplane and sum up
        :param ctx:
        :param input: (B, C, H, W)
        :param grid: (B, h, w, 2)
        :param mode: 'bilinear' or 'nearest'
        :param padding_mode:
        :param align_corners:
        :return: sampled features (B, n)
        """
        mode_enum = GRID_SAMPLE_INTERPOLATION_MODES[mode]
        padding_mode_enum = GRID_SAMPLE_PADDING_MODES[padding_mode]
        sampled = triplane_sampler_cuda.forward(input, grid, mode_enum, padding_mode_enum,
                                                align_corners)
        ctx.save_for_backward(input, grid)
        ctx.mode_enum = mode_enum
        ctx.padding_mode_enum = padding_mode_enum
        ctx.align_corners = align_corners
        ctx.output_mask = output_mask
        return sampled

    @staticmethod
    def backward(ctx, grad_output):
        """
        backward for grid_sample
        :param ctx:
        :param grad_output:
        :return:
        """
        input, grid = ctx.saved_tensors
        mode_enum = ctx.mode_enum
        padding_mode_enum = ctx.padding_mode_enum
        align_corners = ctx.align_corners
        output_mask = ctx.output_mask

        grad_input, grad_grid = triplane_sampler_cuda.backward(
            grad_output, input, grid, mode_enum, padding_mode_enum, align_corners, output_mask)
        return grad_input, grad_grid, None, None, None, None


def grid_sampler(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    output_mask = (input.requires_grad, grid.requires_grad)
    return GridSamplerFunction.apply(input, grid, mode, padding_mode, align_corners, output_mask)


class TriplaneSamplerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, triplane, n_parts, triplane_coord_3d, index):
        """
        grid_sample feature and probability for each part from triplane and sum up
        :param ctx:
        :param triplane: (B, 32 + n_parts, H, W)
        :param n_parts: number of parts
        :param triplane_coord_3d: (N, 3) where N is total the number of points to grid_sample
        :param index: (B, n) where n is the number of sampled points
        :return: sampled features (B, n)
        """
        sampled, part_prob, part_feature = triplane_sampler_cuda.forward(triplane, n_parts, triplane_coord_3d, index)
        ctx.save_for_backward(part_prob, part_feature)
        return sampled

    @staticmethod
    def backward(ctx, grad_sampled):
        """

        :param ctx:
        :param grad_sampled:
        :return:
        """
        d_triplane = triplane_sampler_cuda.backward(
            grad_sampled.contiguous(), *ctx.saved_tensors)
        return d_triplane, None, None, None


class TriplaneSampler(nn.Module):
    def __init__(self, n_parts: int, coordinate_scale: float = 3.0, n_coarse: int = 48, n_fine: int = 64):
        super(TriplaneSampler, self).__init__()
        self.n_parts = n_parts
        self.coordinate_scale = coordinate_scale
        self.n_coarse = n_coarse
        self.n_fine = n_fine

    def prepare_inputs(self, coord_3d: torch.Tensor, transformation: torch.Tensor, sampled_depth: torch.Tensor,
                       cube_intersection: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param coord_3d: coordinate in camera coordinate (B, 3, n_rays, n_sample)
        :param transformation: observation to canonical (B, n_parts, 4, 4)
        :param sampled_depth: (B, n_rays, n_coarse)
        :param cube_intersection: intersection of ray and cubes (B, n_rays, n_parts, 2)
        :return:

        """
        _sampled_depth = sampled_depth[:, :, :, None]
        part_existence = ((cube_intersection[:, :, None, :, 0] < _sampled_depth) &
                          (_sampled_depth < cube_intersection[:, :, None, :, 1]))  # (B, n_rays, n_sample, n_parts)

        # TODO: implement cpp version
        batchsize, _, n_rays, n_sample_per_ray = coord_3d.shape

        for i in range(self.n_parts):
            canonical_i = transformation[:, i, :3, :3].matmul(coord_3d) + transformation[:, i, :3, 3:]  # (B, 3, n
            part_existence_i = (-1 / self.coordinate_scale < canonical_i) & (canonical_i < 1 / self.coordinate_scale)
            part_existence_i = part_existence_i.all(dim=1)  # (B, n)
            part_existence[:, :, i] = part_existence_i
        exist_part_idx = torch.where(part_existence.reshape(-1))  # (N,)
        point_idx = part_existence.sum(dim=2).reshape(-1).cumsum(dim=0).reshape(batchsize, n_points_sampled)  # (B, n)

    def forward(self, triplane, triplane_coord_3d, index):
        return TriplaneSamplerFunction.apply(triplane, self.n_parts, triplane_coord_3d, index)

    def inference(self, triplane, triplane_coord_3d, index):
        return triplane_sampler_cuda.inference(triplane, self.n_parts, triplane_coord_3d, index)


triplane_sampler = TriplaneSamplerFunction.apply

if __name__ == "__main__":
    cuda_device = 'cuda'
    batch_size = 16
    input_features = 32
    state_size = 128

    X = torch.randn(batch_size, input_features, device=cuda_device)
    h = torch.randn(batch_size, state_size, device=cuda_device)
    C = torch.randn(batch_size, state_size, device=cuda_device)

    forward = 0
    backward = 0
    n = 20000
    for _ in tqdm(range(n)):
        torch.cuda.synchronize()
        start = time.time()
        new_h, new_C = triplane_sampler(X, (h, C))
        torch.cuda.synchronize()
        forward += time.time() - start
        torch.cuda.synchronize()
        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        torch.cuda.synchronize()
        backward += time.time() - start

    print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward / n * 1e6, backward / n * 1e6))
