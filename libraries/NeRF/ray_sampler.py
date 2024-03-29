from typing import Tuple

import torch
import torch.nn.functional as F


def mask_based_sampler(mask: torch.Tensor, ray_batchsize: int
                       ) -> Tuple[torch.tensor, torch.tensor]:
    """sample rays from entire image

    Args:
        mask: foreground mask, (B, img_size, img_size)
        ray_batchsize: batchsize of ray per image
        batchsize: batchsize of minibatch

    Returns:
        grid: normalized coordinates used for torch.nn.functional.grid_sample, (B, patch_size, patch_size, 2)
        homo_img: homogeneous image coordinate, (B, 1, 3, patch_size ** 2)

    """
    batchsize, h, w = mask.shape

    # expand mask
    pad_size = 64  # h // 8
    mask = F.max_pool2d(mask.float(), pad_size * 2 + 1, stride=1, padding=pad_size)
    mask = mask.reshape(batchsize, h * w)

    # random sample
    mask = mask + torch.empty_like(mask).uniform_()
    ray_idx = torch.topk(mask, ray_batchsize, dim=1, sorted=False)[1]
    x = ray_idx % w  # (B, ray_bs)
    y = ray_idx // w  # (B, ray_bs)

    rays = torch.stack([x, y], dim=2) + 0.5

    rays = rays.permute(0, 2, 1)
    homo_img = torch.cat([rays, torch.ones(batchsize, 1, ray_batchsize, device="cuda")], dim=1)  # B x 3 x n
    homo_img = homo_img.reshape(batchsize, 1, 3, -1)
    return ray_idx, homo_img


def whole_image_grid_ray_sampler(render_size: int, patch_size: int, batchsize: int
                                 ) -> Tuple[torch.tensor, torch.tensor]:
    """sample rays from entire image

    Args:
        render_size:
        patch_size:
        batchsize:

    Returns:
        grid: normalized coordinates used for torch.nn.functional.grid_sample, (B, patch_size, patch_size, 2)
        homo_img: homogeneous image coordinate, (B, 1, 3, patch_size ** 2)

    """
    y, x = torch.meshgrid([torch.arange(patch_size, device="cuda"),
                           torch.arange(patch_size, device="cuda")])
    rays = torch.stack([x, y], dim=2)[None]
    rays = render_size * (rays + 0.5) / patch_size
    rays = rays.repeat(batchsize, 1, 1, 1)  # B x patch_size x patch_size x 2

    grid = rays / (render_size / 2) - 1  # [-1, 1]

    rays = rays.reshape(batchsize, -1, 2).permute(0, 2, 1)
    homo_img = torch.cat([rays, torch.ones(batchsize, 1, patch_size ** 2, device="cuda")], dim=1)  # B x 3 x n
    homo_img = homo_img.reshape(batchsize, 1, 3, -1)
    return grid, homo_img
