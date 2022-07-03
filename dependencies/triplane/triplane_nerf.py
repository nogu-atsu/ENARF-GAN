import sys
from typing import Union, List, Optional, Dict

import torch
from torch import nn

from dependencies.NeRF.base import NeRFBase
from dependencies.NeRF.net import StyledMLP
from dependencies.NeRF.utils import StyledConv1d, positional_encoding, in_cube, to_local
from dependencies.custom_stylegan2.net import EqualConv1d
from dependencies.triplane.sampling import sample_feature

sys.path.append("dependencies/stylegan2_ada_pytorch")
import dnnlib


def prepare_triplane_generator(z_dim, w_dim, out_channels, c_dim=0):
    G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=z_dim, w_dim=w_dim,
                               mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict(use_noise=False))
    G_kwargs.synthesis_kwargs.channel_base = 32768
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = 8
    G_kwargs.synthesis_kwargs.num_fp16_res = 0
    G_kwargs.synthesis_kwargs.conv_clamp = None

    g_common_kwargs = dict(c_dim=c_dim,
                           img_resolution=256, img_channels=out_channels)
    gen = dnnlib.util.construct_class_by_name(**G_kwargs, **g_common_kwargs)
    return gen


def calc_density_and_color_from_feature(self, feature, z_rend, ray_direction):
    if self.view_dependent:
        density = self.density_fc(feature, z_rend)  # (B, 1, n)
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
        color_density = self.mlp(feature, z_rend)  # (B, 4, n)
        color, density = color_density[:, :3], color_density[:, 3:]
        color = torch.tanh(color)
    return density, color


class TriPlaneNeRF(NeRFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, view_dependent: bool = False):
        self.tri_plane_based = True
        self.w_dim = 512
        self.feat_dim = 32
        self.no_selector = config.no_selector
        super(TriPlaneNeRF, self).__init__(config, z_dim, view_dependent)
        self.initialize_network()

    def initialize_network(self):
        if self.config.constant_triplane:
            self.tri_plane = nn.Parameter(torch.zeros(1, 32 * 3, 256, 256))
            self.tri_plane_gen = lambda z, *args, **kwargs: self.tri_plane.expand(z.shape[0], -1, -1, -1)
        else:
            self.tri_plane_gen = self.prepare_stylegan2((self.feat_dim + self.num_bone) * 3)

        if self.view_dependent:
            self.density_fc = StyledConv1d(32, 1, self.z2_dim)
            self.mlp = StyledMLP(32 + 3 * self.num_frequency_for_other * 2, 64, 3, style_dim=self.z2_dim)
        else:
            print("not view dependent")
            self.mlp = StyledMLP(32, 64, 4, style_dim=self.z2_dim)

    @property
    def memory_cost(self):
        m = 0
        for layer in self.children():
            if isinstance(layer, (StyledConv1d, EqualConv1d, StyledMLP)):
                m += layer.memory_cost
        return m

    @property
    def flops(self):
        fl = 0
        for layer in self.children():
            if isinstance(layer, (StyledConv1d, EqualConv1d, StyledMLP)):
                fl += layer.flops

        if self.z_dim > 0:
            fl += self.hidden_size * 2
        if self.use_bone_length:
            fl += self.hidden_size
        fl += self.hidden_size * 2
        return fl

    def prepare_stylegan2(self, in_channels):
        return prepare_triplane_generator(self.z_dim, self.w_dim, in_channels)

    def calc_density_and_color_from_camera_coord_v2(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                    ray_direction: torch.Tensor, model_input: Dict):
        """compute density from positions in camera coordinate

        :param position: (B, 3, n), n is a very large number of points sampled
        :param pose_to_camera:
        :param ray_direction:
        :param model_input: dictionary of model input
        :return: density of input positions
        """
        z, z_rend = model_input["z"], model_input["z_rend"]
        tri_plane_feature, truncation_psi = model_input.get("tri_plane_feature"), model_input["truncation_psi"]

        if self.tri_plane_based:
            if tri_plane_feature is None:
                z = self.compute_tri_plane_feature(z, truncation_psi)
            else:
                z = tri_plane_feature
            self.buffers_tensors["tri_plane_feature"] = z
            if not self.training:
                self.temporal_state["tri_plane_feature"] = z
        # to local and canonical coordinate (challenge: this is heavy (B, n_bone * 3, n))
        local_points, canonical_points = to_local(position, pose_to_camera)

        in_cube_p = in_cube(local_points)  # (B, n_bone, n)
        in_cube_p = in_cube_p * (canonical_points.abs() < 1).all(dim=2)  # (B, n_bone, n)
        density, color = self.backbone(canonical_points, in_cube_p, z, z_rend, ray_direction)
        density *= in_cube_p.any(dim=1, keepdim=True)  # density is 0 if not in cube

        if not self.training:
            self.temporal_state.update({
                "canonical_fine_points": canonical_points,
                "in_cube": in_cube(local_points),
            })
        return density, color

    def backbone(self, p: torch.Tensor, position_validity: torch.Tensor, tri_plane_feature: torch.Tensor,
                 z_rend: torch.Tensor, ray_direction: Optional[torch.Tensor] = None):
        """

        Args:
            p: position in canonical coordinate, (B, n_bone, 3, n)
            position_validity: bool tensor for validity of p, (B, n_bone, n)
            tri_plane_feature:
            z_rend: (B, dim)
            ray_direction: not None if color is view dependent
        Returns:

        """
        # don't support mip-nerf rendering
        assert isinstance(p, torch.Tensor)

        bs, n_bone, _, n = p.shape

        # Make the invalid position outside the range of -1 to 1 (all invalid positions become 2)
        masked_position = p * position_validity[:, :, None] + 2 * ~position_validity[:, :, None]

        feature = sample_feature(tri_plane_feature, masked_position)

        density, color = calc_density_and_color_from_feature(self, feature, z_rend, ray_direction)
        return density, color

    def compute_tri_plane_feature(self, z, truncation_psi=1):
        """
        Generate triplane features with stylegan
        :param z:
        :param truncation_psi:
        :return:
        """
        # generate tri-plane feature conditioned on z
        tri_plane_feature = self.tri_plane_gen(z, truncation_psi=truncation_psi)  # (B, 32 * 3, h, w)
        return tri_plane_feature
