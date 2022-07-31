from typing import Union, List

import numpy as np
import torch

from libraries.custom_stylegan2.net import StyledConv

StyledConv1d = lambda in_channel, out_channel, style_dim, groups=1: StyledConv(in_channel, out_channel, 1, style_dim,
                                                                               use_noise=False, conv_1d=True,
                                                                               groups=groups)


def to_local(points, pose_to_camera):
    """transform points to local coordinate

    Args:
        points:
        pose_to_camera:

    Returns:

    """
    # to local coordinate
    R = pose_to_camera[:, :, :3, :3]  # (B, n_bone, 3, 3)
    inv_R = R.permute(0, 1, 3, 2)
    t = pose_to_camera[:, :, :3, 3:]  # (B, n_bone, 3, 1)
    local_points = torch.matmul(inv_R, points[:, None] - t)  # (B, n_bone, 3, n*Nc)

    # reshape local
    bs, n_bone, _, n = local_points.shape
    local_points = local_points.reshape(bs, n_bone * 3, n)
    return local_points


def in_cube(p: torch.Tensor):
    # whether the positions are in the cube [-1, 1]^3
    # :param p: b x groups * 3 x n (n = num_of_ray * points_on_ray)
    if p.shape[1] == 3:
        inside = (p.abs() <= 1).all(dim=1, keepdim=True)  # ? x 1 x ???
        return inside
    b, _, n = p.shape
    inside = (p.reshape(b, -1, 3, n).abs() <= 1).all(dim=2)
    return inside  # b x groups x 1 x n


def multi_part_positional_encoding(value: Union[List, torch.tensor], num_frequency: int, num_bone: int):
    """
    positional encoding for group conv
    :param value: b x -1 x n
    :param num_frequency: L in NeRF paper
    :param num_bone: num_bone for positional encoding
    :return:
    """
    if isinstance(value, list):  # mip-nerf positional encoding
        val, diag_sigma = value
        b, _, n = val.shape
        val, diag_sigma = val.reshape(b * num_bone, -1, n), diag_sigma.reshape(b * num_bone, -1, n)
        gamma_p = mip_nerf_positional_encoding(val, diag_sigma, num_frequency)
    else:
        val = value
        b, _, n = val.shape
        val = val.reshape(b * num_bone, -1, n)
        gamma_p = positional_encoding(val, num_frequency, cos_first=False, cat_dim=1)

    gamma_p = gamma_p.reshape(b, -1, n)

    # mask outsize [-1, 1]
    mask = (val.reshape(b, num_bone, -1, n).abs() > 1).float().sum(dim=2, keepdim=True) >= 1
    mask = mask.float().repeat(1, 1, gamma_p.shape[1] // num_bone, 1)
    mask = mask.reshape(gamma_p.shape)
    return gamma_p * (1 - mask)  # B x (groups * ? * L * 2) x n


def positional_encoding(x: torch.Tensor, num_frequency: int, cos_first=True, cat_dim=2) -> torch.Tensor:
    """
    positional encoding
    :param x: (B, dim, n)
    :param num_frequency: L in nerf paper
    :return:(B, dim * n_freq * 2)
    """
    bs, dim, n = x.shape
    x = x.unsqueeze(cat_dim) * 2 ** torch.arange(num_frequency,
                                                 device=x.device)[None, :, None].unsqueeze(3 - cat_dim) * np.pi
    if cos_first:
        encoded = torch.cat([torch.cos(x), torch.sin(x)], dim=cat_dim)
    else:
        encoded = torch.cat([torch.sin(x), torch.cos(x)], dim=cat_dim)
    return encoded.reshape(bs, -1, n)


def mip_nerf_positional_encoding(value: torch.Tensor, diag_sigma: torch.Tensor, num_frequency: int):
    """
    positional encoding of mip-nerf
    :param value: (B, dim ,n)
    :param diag_sigma: (B, dim, n)
    :param num_frequency: L in NeRF paper
    :return:
    """
    bs, dim, n = value.shape
    gamma_p = positional_encoding(value, num_frequency, cos_first=False, cat_dim=1)
    diag_sigma = diag_sigma[:, None, :, :] * 4 ** torch.arange(num_frequency,
                                                               device=diag_sigma.device)[None, :, None, None] * np.pi
    diag_sigma = torch.exp(-diag_sigma / 2)
    diag_sigma = torch.cat([diag_sigma, diag_sigma], dim=1).reshape(bs, -1, n)
    gamma_p = gamma_p * diag_sigma
    return gamma_p
