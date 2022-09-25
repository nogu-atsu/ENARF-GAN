import time

import torch
import triplane_sampler_cuda
from tqdm import tqdm

GRID_SAMPLE_INTERPOLATION_MODES = {
    "bilinear": 0,
    "nearest": 1,
}

GRID_SAMPLE_PADDING_MODES = {
    "zeros": 0,
    "border": 1,
    "reflection": 2,
}


class TriplaneSamplerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='zeros', align_corners=False, output_mask=(True, True)):
        """
        grid_sample feature and probability for each part from triplane and sum up
        :param ctx:
        :param input: (B, C, H, W)
        :param grid: (B, h, w, 3)
        :param mode: 'bilinear' or 'nearest'
        :param padding_mode:
        :param align_corners:
        :return: sampled features (B, n)
        """
        mode_enum = GRID_SAMPLE_INTERPOLATION_MODES[mode]
        padding_mode_enum = GRID_SAMPLE_PADDING_MODES[padding_mode]
        sampled = triplane_sampler_cuda.triplane_sampler_forward(input, grid, mode_enum, padding_mode_enum,
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

        grad_input, grad_grid = triplane_sampler_cuda.triplane_sampler_backward(
            grad_output, input, grid, mode_enum, padding_mode_enum, align_corners, output_mask)
        if output_mask[0]:
            grad_input = None
        if output_mask[1]:
            grad_grid = None
        return grad_input, grad_grid, None, None, None, None


def triplane_sampler(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    output_mask = (input.requires_grad, grid.requires_grad)
    return TriplaneSamplerFunction.apply(input, grid, mode, padding_mode, align_corners, output_mask)


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
