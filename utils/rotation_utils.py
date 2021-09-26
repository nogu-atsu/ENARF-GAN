import numpy as np
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_matrix(theta: torch.tensor):
    batchsize = theta.shape[0]
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    R = torch.stack([c, z, -s, z,
                     z, o, z, z,
                     s, z, c, z,
                     z, z, z, o], dim=-1)  # (B, 16)
    R = R.reshape(batchsize, 4, 4)
    return R


def rotate_pose_randomly(pose_3d: torch.tensor):
    batchsize = pose_3d.shape[0]
    rotate_angle = pose_3d.new_empty((batchsize,)).uniform_(0, 2 * np.pi)
    R = rotation_matrix(rotate_angle)

    rotated_pose_3d = rotate_pose(pose_3d, R)
    return rotated_pose_3d


def rotate_pose_by_angle(pose_3d: torch.tensor, angle):
    R = rotation_matrix(angle)
    rotated_pose_3d = rotate_pose(pose_3d, R)
    return rotated_pose_3d


def rotate_pose(pose_3d: torch.tensor, R: torch.tensor):
    zeros_33 = torch.zeros(pose_3d.shape[0], 3, 3, device=R.device, dtype=torch.float)
    center = torch.cat([zeros_33, pose_3d[:, :, :3, 3:].mean(dim=1)], dim=-1)
    zeros_14 = torch.zeros(pose_3d.shape[0], 1, 4, device=R.device, dtype=torch.float)
    center = torch.cat([center, zeros_14], dim=1)[:, None]
    pose_camera_theta = torch.matmul(R[:, None], (pose_3d - center)) + center
    return pose_camera_theta
