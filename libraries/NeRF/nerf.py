from typing import Union, List, Optional, Dict

import torch

from libraries.NeRF.base import NeRFBase
from libraries.NeRF.net import StyledMLP, MLP
from libraries.NeRF.utils import StyledConv1d, positional_encoding, in_cube, to_local


def calc_density_and_color_from_feature(self, feature, z_rend, ray_direction):
    density = self.density_fc(feature, z_rend)  # (B, 1, n)
    if self.view_dependent:
        if ray_direction is None:
            color = None
        else:
            ray_direction = positional_encoding(ray_direction, self.num_frequency_for_other)
            ray_direction = torch.repeat_interleave(ray_direction,
                                                    feature.shape[-1] // ray_direction.shape[-1],
                                                    dim=2)
            color = self.mlp(torch.cat([feature, ray_direction], dim=1), z_rend)  # (B, 3, n)
            color = torch.tanh(color)
    else:
        color = self.mlp(feature, z_rend)  # (B, 4, n)
        color = torch.tanh(color)

    density = self.density_activation(density)
    return density, color


class MLPNeRF(NeRFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, view_dependent: bool = True):
        self.tri_plane_based = False
        super(MLPNeRF, self).__init__(config, z_dim, view_dependent)
        self.initialize_network()

    def initialize_network(self):
        hidden_size = self.hidden_size

        self.density_mlp = MLP(3 * self.num_frequency_for_position * 2, hidden_size,
                               hidden_size, num_layers=8, skips=(4,))

        self.density_fc = StyledConv1d(self.hidden_size, 1, self.z2_dim)
        if self.view_dependent:
            self.mlp = StyledMLP(self.hidden_size + 3 * self.num_frequency_for_other * 2, self.hidden_size // 2,
                                 3, style_dim=self.z2_dim)
        else:
            self.mlp = StyledMLP(self.hidden_size, self.hidden_size // 2, 3, style_dim=self.z2_dim)

    def calc_density_and_color_from_camera_coord_v2(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                    ray_direction: torch.Tensor, model_input: Dict = {}):
        """compute density from positions in camera coordinate

        :param position:
        :param pose_to_camera:
        :param ray_direction:
        :param model_input:
        :return: density of input positions
        """
        z, z_rend = model_input.get("z"), model_input.get("z_rend")

        local_points = to_local(position, pose_to_camera)

        in_cube_p = in_cube(local_points)  # (B, n_bone, n)
        density, color = self.backbone(local_points, in_cube_p, z, z_rend, ray_direction)
        density *= in_cube_p.any(dim=1, keepdim=True)
        return density, color

    def backbone(self, p: torch.Tensor, position_validity: torch.Tensor, z: torch.Tensor,
                 z_rend: torch.Tensor, ray_direction: Optional[torch.Tensor] = None):
        """

        Args:
            p: position in local coordinate, (B, n_bone, 3, n)
            position_validity: bool tensor for validity of p, (B, n_bone, n)
            z: (B, dim)
            z_rend: (B, dim)
            ray_direction: not None if color is view dependent
        Returns:

        """
        assert isinstance(p, torch.Tensor)
        encoded_p = positional_encoding(p, self.num_frequency_for_position)
        feature = self.density_mlp(encoded_p)
        density, color = calc_density_and_color_from_feature(self, feature, z_rend, ray_direction)
        return density, color
