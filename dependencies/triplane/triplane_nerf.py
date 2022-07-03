import sys
from typing import Union, List, Optional, Dict

import torch
import torch.nn.functional as F
from torch import nn

from dependencies.NeRF.base import NeRFBase
from dependencies.NeRF.net import StyledMLP
from dependencies.NeRF.utils import StyledConv1d, encode, positional_encoding, in_cube
from dependencies.custom_stylegan2.net import EqualConv1d
from dependencies.triplane.sampling import sample_feature, sample_triplane_part_prob, sample_weighted_feature_v2

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


class TriPlaneNeRF(NeRFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256,
                 view_dependent: bool = False):
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
        elif self.config.constant_trimask:
            self.generator = self.prepare_stylegan2(self.feat_dim * 3)
            lr_mul = self.config.constant_trimask_lr_mul
            self.tri_plane = nn.Parameter(torch.zeros(1, self.num_bone * 3, 256, 256) / lr_mul)
            self.tri_plane_gen = lambda z, *args, **kwargs: torch.cat(
                [self.generator(z, *args, **kwargs),
                 self.tri_plane.expand(z.shape[0], -1, -1, -1) * lr_mul], dim=1)
        elif self.config.deformation_field:
            self.tri_plane = nn.Parameter(torch.zeros(1, 32 * 3 + self.num_bone * 3, 256, 256))
            self.flow_generator = self.prepare_stylegan2(2 * 3)

            def warp(z, *args, **kwargs):
                bs = z.shape[0]
                tri_plane_size = 256
                flow = self.flow_generator(z, *args, **kwargs)
                flow = flow.reshape(bs * 3, 2, tri_plane_size, tri_plane_size).permute(0, 2, 3, 1)
                arange = torch.arange(tri_plane_size, device=z.device)
                grid = torch.stack(torch.meshgrid(arange, arange)[::-1], dim=2) + 0.5
                grid = (grid + flow) / 128 - 1  # warped grid in [-1, 1], (3B, 256, 256, 2)
                tri_plane = self.tri_plane.expand(z.shape[0], -1, -1, -1)
                warped_feature = F.grid_sample(tri_plane[:, :32 * 3].reshape(bs * 3, 32, tri_plane_size,
                                                                             tri_plane_size), grid)
                warped_feature = warped_feature.reshape(bs, 32 * 3, tri_plane_size, tri_plane_size)
                tri_plane = torch.cat([warped_feature, tri_plane[:, 32 * 3:]], dim=1)
                return tri_plane

            self.tri_plane_gen = warp
        elif self.config.selector_mlp:
            self.generator = self.prepare_stylegan2(self.feat_dim * 3)
            self.tri_plane_gen = lambda z, *args, **kwargs: torch.cat(
                [self.generator(z, *args, **kwargs),
                 z.new_zeros(z.shape[0], self.num_bone * 3, 256, 256)], dim=1)

            self.selector = nn.Sequential(EqualConv1d(3 * self.num_bone * self.num_frequency_for_position * 2,
                                                      10 * self.num_bone, 1, groups=self.num_bone),
                                          nn.ReLU(inplace=True),
                                          EqualConv1d(10 * self.num_bone, self.num_bone, 1,
                                                      groups=self.num_bone))
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

    def sample_weighted_feature_v2(self, tri_plane_features: torch.Tensor, position: torch.Tensor,
                                   weight: torch.Tensor, position_validity: torch.Tensor, padding_value: float = 0):
        # only compute necessary elements
        batchsize, n_bone, n = position_validity.shape
        _, ch, tri_size, _ = tri_plane_features.shape

        # place tri-planes side by side to form a single tri-plane (quite tricky)
        feature_padded = F.pad(tri_plane_features, (0, 1))  # (B, ch, 256, 257)
        feature_padded = feature_padded.permute(1, 2, 0, 3).reshape(1, ch, tri_size, (tri_size + 1) * batchsize)

        # gather valid rays
        position_validity = position_validity.reshape(-1)
        assert position_validity.dtype == torch.bool
        valid_args = torch.where(position_validity)[0]  # (num_valid, )
        num_valid = valid_args.shape[0]

        if num_valid > 0:  # num_valid is 3e7 for zju dataset
            position_perm = position.permute(2, 0, 1, 3).reshape(3, batchsize * n_bone * n)  # (3, B * n_bone * n)
            valid_positions = torch.gather(position_perm, dim=1,
                                           index=valid_args[None].expand(3, -1))[None]  # (1, 3, num_valid)
            # challenge: this is very heavy
            value = sample_feature(feature_padded, valid_positions, clamp_mask=self.config.clamp_mask,
                                   batch_idx=valid_args // (n_bone * n))  # (1, 32, num_valid)
            # gather weight
            weight = torch.gather(weight.reshape(-1), dim=0, index=valid_args)

            # * weight
            value = value * weight[None, None]  # (1, 32, num_valid)

            # memory efficient
            output = torch.zeros(self.feat_dim, batchsize * n, device=position.device, dtype=torch.float32)
            scatter_idx = valid_args // (n_bone * n) * n + valid_args % n
            output.scatter_add_(dim=1, index=scatter_idx[None].expand(32, -1), src=value.squeeze(0))
            output = output.reshape(self.feat_dim, batchsize, n).permute(1, 0, 2)
            output = output.contiguous()
        else:
            output = torch.zeros(batchsize, self.feat_dim, n, device=position.device, dtype=torch.float32)
        return output

    def calc_weight(self, tri_plane_weights: torch.Tensor, position: torch.Tensor, position_validity: torch.Tensor,
                    mode="prod"):
        """
        return part prob. MLP/triplane/constant
        :param tri_plane_weights:
        :param position:
        :param position_validity:
        :param mode:
        :return:
        """
        bs, n_bone, _, n = position.shape
        if self.no_selector:
            weight = torch.ones(bs, n_bone, n, device=position.device) / n_bone

        elif hasattr(self, "selector"):  # use selector
            position = position.reshape(bs, n_bone * 3, n)
            encoded_p = encode(position, self.num_frequency_for_position, self.num_bone)
            h = self.selector(encoded_p)
            weight = torch.softmax(h, dim=1)  # (B, n_bone, n)
        else:  # tri-plane based
            weight = sample_triplane_part_prob(tri_plane_weights, position, position_validity, mode=mode,
                                               clamp_mask=self.config.clamp_mask)

        return weight

    def to_local(self, points, pose_to_camera):
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
        local_points = torch.matmul(inv_R, points[:, None] - t)  # (B, n_bone, 3, n)

        # reshape local
        bs, n_bone, _, n = local_points.shape
        local_points = local_points.reshape(bs, n_bone * 3, n)
        return local_points

    def calc_density_and_color_from_camera_coord(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                 bone_length: torch.Tensor, z, z_rend, ray_direction):
        """compute density from positions in camera coordinate

        :param position: (B, 3, n), n is a very large number of points sampled
        :param pose_to_camera:
        :param bone_length:
        :param z:
        :param z_rend:
        :param ray_direction:
        :return: density of input positions
        """
        # to local and canonical coordinate (challenge: this is heavy (B, n_bone * 3, n))
        local_points, canonical_points = self.to_local_and_canonical(position, pose_to_camera, bone_length)

        in_cube_p = in_cube(local_points)  # (B, n_bone, n)
        in_cube_p = in_cube_p * (canonical_points.abs() < 1).all(dim=2)  # (B, n_bone, n)
        density, color = self.backbone(canonical_points, in_cube_p, z, z_rend, bone_length, "weight_feature",
                                       ray_direction)
        density *= in_cube_p.any(dim=1, keepdim=True)  # density is 0 if not in cube

        if not self.training:
            self.temporal_state.update({
                "canonical_fine_points": canonical_points,
                "in_cube": in_cube(local_points),
            })
        return density, color

    def calc_density_and_color_from_camera_coord_v2(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                    ray_direction: torch.Tensor, model_input: Dict):
        """compute density from positions in camera coordinate

        :param position: (B, 3, n), n is a very large number of points sampled
        :param pose_to_camera:
        :param ray_direction:
        :param model_input: dictionary of model input
        :return: density of input positions
        """
        bone_length, z, z_rend = model_input["bone_length"], model_input["z"], model_input["z_rend"]
        tri_plane_feature, truncation_psi = model_input.get("tri_plane_feature"), model_input["truncation_psi"]

        if self.tri_plane_based:
            if tri_plane_feature is None:
                z = self.compute_tri_plane_feature(z, bone_length, truncation_psi)
            else:
                z = tri_plane_feature
            self.buffers_tensors["tri_plane_feature"] = z
            if not self.training:
                self.temporal_state["tri_plane_feature"] = z
        # to local and canonical coordinate (challenge: this is heavy (B, n_bone * 3, n))
        local_points, canonical_points = self.to_local_and_canonical(position, pose_to_camera, bone_length)

        in_cube_p = in_cube(local_points)  # (B, n_bone, n)
        in_cube_p = in_cube_p * (canonical_points.abs() < 1).all(dim=2)  # (B, n_bone, n)
        density, color = self.backbone(canonical_points, in_cube_p, z, z_rend, bone_length, "weight_feature",
                                       ray_direction)
        density *= in_cube_p.any(dim=1, keepdim=True)  # density is 0 if not in cube

        if not self.training:
            self.temporal_state.update({
                "canonical_fine_points": canonical_points,
                "in_cube": in_cube(local_points),
            })
        return density, color

    def backbone(self, p: torch.Tensor, position_validity: torch.Tensor, tri_plane_feature: torch.Tensor,
                 z_rend: torch.Tensor, bone_length: torch.Tensor, mode: str = "weight_feature",
                 ray_direction: Optional[torch.Tensor] = None):
        """

        Args:
            p: position in canonical coordinate, (B, n_bone, 3, n)
            position_validity: bool tensor for validity of p, (B, n_bone, n)
            tri_plane_feature:
            z_rend: (B, dim)
            bone_length: (B, n_bone)
            mode: "weight_feature" or "weight_position"
            ray_direction: not None if color is view dependent
        Returns:

        """
        # don't support mip-nerf rendering
        assert isinstance(p, torch.Tensor)
        assert bone_length is not None
        assert mode in ["weight_position", "weight_feature"]

        bs, n_bone, _, n = p.shape

        # Make the invalid position outside the range of -1 to 1 (all invalid positions become 2)
        masked_position = p * position_validity[:, :, None] + 2 * ~position_validity[:, :, None]

        weight = self.calc_weight(tri_plane_feature[:, 32 * 3:].reshape(bs * n_bone, 3, 256, 256),
                                  masked_position, position_validity)  # (bs, n_bone, n)

        if not self.training:
            self.temporal_state.update({
                "weight": weight,
            })
        if weight.requires_grad:
            if not hasattr(self, "buffers_tensors"):
                self.buffers_tensors = {}

            self.buffers_tensors["mask_weight"] = weight

        # default mode is "weight_feature"
        # weighted sum of tri-plane features
        if mode == "weight_feature":
            feature = sample_weighted_feature_v2(self.feat_dim, tri_plane_feature[:, :32 * 3], masked_position,
                                                 weight, position_validity,
                                                 clamp_mask=self.config.clamp_mask)  # (B, 32, n)
        # canonical position based
        elif mode == "weight_position":
            weighted_position_validity = position_validity.any(dim=1)[:, None]
            weighted_position = (p * weight[:, :, None]).sum(dim=1)  # (bs, 3, n)
            # Make the invalid position outside the range of -1 to 1 (all invalid positions become 2)
            weighted_position = weighted_position * weighted_position_validity + 2 * ~weighted_position_validity
            feature = sample_feature(tri_plane_feature[:, :32 * 3], weighted_position,
                                     clamp_mask=self.config.clamp_mask, )  # (B, 32, n)
        else:
            raise ValueError()

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
        if self.config.multiply_density_with_triplane_wieght:
            density = self.density_activation(density) * (10 * weight.max(dim=1, keepdim=True)[0])
        else:
            density = self.density_activation(density) * 10
        return density, color

    def compute_tri_plane_feature(self, z, bone_length, truncation_psi=1):
        """
        Generate triplane features with stylegan
        :param z:
        :param bone_length:
        :param truncation_psi:
        :return:
        """
        # generate tri-plane feature conditioned on z and bone_length
        encoded_length = encode(bone_length, self.num_frequency_for_other, num_bone=self.num_bone_param)
        tri_plane_feature = self.tri_plane_gen(z, encoded_length[:, :, 0],
                                               truncation_psi=truncation_psi)  # (B, (32 + n_bone) * 3, h, w)
        return tri_plane_feature
