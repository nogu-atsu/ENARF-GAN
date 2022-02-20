import sys
import warnings
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from NARF.models.activation import MyReLU
from NARF.models.model_utils import in_cube
from NARF.models.nerf_model import NeRF
from models.stylegan import StyledConv, EqualConv1d

StyledConv1d = lambda in_channel, out_channel, style_dim, groups=1: StyledConv(in_channel, out_channel, 1, style_dim,
                                                                               use_noise=False, conv_1d=True,
                                                                               groups=groups)


def encode(value: Union[List, torch.tensor], num_frequency: int, num_bone: int):
    """
    positional encoding for group conv
    :param value: b x -1 x n
    :param num_frequency: L in NeRF paper
    :param num_bone: num_bone for positional encoding
    :return:
    """
    # with autocast(enabled=False):
    if isinstance(value, list):
        val, diag_sigma = value
    else:
        val = value
        diag_sigma = None
    b, _, n = val.shape
    values = [2 ** i * val.reshape(b, num_bone, -1, n) * np.pi for i in range(num_frequency)]
    values = torch.cat(values, dim=2)
    gamma_p = torch.cat([torch.sin(values), torch.cos(values)], dim=2)
    if diag_sigma is not None:
        diag_sigmas = [4 ** i * diag_sigma.reshape(b, num_bone, -1, n) * np.pi for i in range(num_frequency)] * 2
        diag_sigmas = torch.cat(diag_sigmas, dim=2)
        gamma_p = gamma_p * torch.exp(-diag_sigmas / 2)
    gamma_p = gamma_p.reshape(b, -1, n)
    # mask outsize [-1, 1]
    mask = (val.reshape(b, num_bone, -1, n).abs() > 1).float().sum(dim=2, keepdim=True) >= 1
    mask = mask.float().repeat(1, 1, gamma_p.shape[1] // num_bone, 1)
    mask = mask.reshape(gamma_p.shape)
    return gamma_p * (1 - mask)  # B x (groups * ? * L * 2) x n


def positional_encoding(x: torch.Tensor, num_frequency: int) -> torch.Tensor:
    """
    positional encoding
    :param x: (B, dim, n)
    :param num_frequency: L in nerf paper
    :return:(B, dim * n_freq * 2)
    """
    bs, dim, n = x.shape
    x = x[:, :, None, :] * 2 ** torch.arange(num_frequency, device=x.device)[:, None] * np.pi
    encoded = torch.cat([torch.cos(x), torch.sin(x)], dim=2)
    return encoded.reshape(bs, -1, n)
