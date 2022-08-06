from typing import Tuple

import torch
from torch import nn

from libraries.NeRF.utils import StyledConv1d
from libraries.custom_stylegan2.net import EqualConv1d


class StyledMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, style_dim=512, num_layers=3):
        super(StyledMLP, self).__init__()
        layers = [StyledConv1d(in_dim, hidden_dim, style_dim)]

        for i in range(num_layers - 2):
            layers.append(StyledConv1d(hidden_dim, hidden_dim, style_dim))

        layers.append(StyledConv1d(hidden_dim, out_dim, style_dim))

        self.layers = nn.ModuleList(layers)
        self.hidden_dim = hidden_dim

    def forward(self, x, z):
        h = x
        for l in self.layers:
            h = l(h, z)
        return h


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3, skips: Tuple = ()):
        super(MLP, self).__init__()
        self.skips = skips
        layers = [EqualConv1d(in_dim, hidden_dim, 1)]

        for i in range(1, num_layers - 1):
            _in_channel = in_dim + hidden_dim if i in skips else hidden_dim
            layers.append(EqualConv1d(_in_channel, hidden_dim, 1))

        layers.append(EqualConv1d(hidden_dim, out_dim, 1))

        self.layers = nn.ModuleList(layers)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, l in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, x], dim=1)
            h = l(h)
        return h
