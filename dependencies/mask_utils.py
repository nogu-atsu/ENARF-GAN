from typing import List

import torch


def create_bone_mask(parent: List, pose_3d: torch.tensor, size: int, intrinsic: torch.tensor,
                     thickness: float = 1.5) -> torch.tensor:
    device = pose_3d.device
    batchsize = pose_3d.shape[0]

    # pose 2d
    pose_translation = pose_3d[:, :, :3, 3:]
    pose_2d = torch.matmul(intrinsic, pose_translation)
    pose_2d = pose_2d[:, :, :2, 0] / pose_2d[:, :, 2:, 0]

    # draw bones
    a = pose_2d[:, 1:]  # end point, (B, num_bone, 2)
    b = pose_2d[:, parent[1:]]  # start point, (B, num_bone, 2)

    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device))
    c = torch.stack([x, y], dim=2).reshape(-1, 2)  # xy coordinate of each pixel

    ab = b - a  # (B, num_bone, 2)
    ac = c[None, None] - a[:, :, None]  # (B, num_bone, size**2, 2)
    acab = torch.matmul(ac, ab[:, :, :, None]).squeeze(3)  # (B, num_bone, size**2)

    abab = (ab ** 2).sum(dim=2, keepdim=True)  # (B, num_bone, 1)
    acac = (ac ** 2).sum(dim=3)  # (B, num_bone, size**2)
    mask = (0 <= acab) * (acab <= abab) * (acab ** 2 >= abab * (acac - thickness ** 2)) * (abab > 1e-8)

    mask = torch.clamp(mask.sum(dim=1), 0, 1).reshape(batchsize, size, size)
    return mask.float()
