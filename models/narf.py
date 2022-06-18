import sys
from typing import Union, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dependencies.NARF.base import NARFBase
from dependencies.NeRF_net.net import StyledMLP, MLP
from dependencies.NeRF_net.utils import StyledConv1d, encode, positional_encoding, in_cube
from dependencies.custom_stylegan2.net import EqualConv1d
from dependencies.triplane.sampling import sample_feature, sample_triplane_part_prob, sample_weighted_feature_v2

sys.path.append("dependencies/stylegan2_ada_pytorch")
import dnnlib


class TriPlaneNARF(NARFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=True, parent=None, num_bone_param=None, view_dependent: bool = False):
        assert bone_length
        self.tri_plane_based = True
        self.w_dim = 512
        self.feat_dim = 32
        self.no_selector = config.no_selector
        super(TriPlaneNARF, self).__init__(config, z_dim, num_bone, bone_length, parent, num_bone_param, view_dependent)

        # self.fc_bone_length = torch.jit.script(
        #     StyledConv1d(self.num_frequency_for_other * 2 * self.num_bone_param,
        #                  self.z_dim, self.z_dim))

        if self.config.constant_triplane:
            self.tri_plane = nn.Parameter(torch.zeros(1, 32 * 3 + self.num_bone * 3, 256, 256))
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

        if view_dependent:
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
        G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=self.z_dim, w_dim=self.w_dim,
                                   mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict(use_noise=False))
        G_kwargs.synthesis_kwargs.channel_base = 32768
        G_kwargs.synthesis_kwargs.channel_max = 512
        G_kwargs.mapping_kwargs.num_layers = 8
        G_kwargs.synthesis_kwargs.num_fp16_res = 0
        G_kwargs.synthesis_kwargs.conv_clamp = None

        g_common_kwargs = dict(c_dim=self.num_frequency_for_other * 2 * self.num_bone_param,
                               img_resolution=256, img_channels=in_channels)
        gen = dnnlib.util.construct_class_by_name(**G_kwargs, **g_common_kwargs)
        return gen

    def register_canonical_pose(self, pose: np.ndarray) -> None:
        """ register canonical pose.

        Args:
            pose: array of (24, 4, 4)

        Returns:

        """
        assert self.origin_location in ["center", "center_fixed", "center+head"]
        coordinate = pose[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parent_id[1:]], axis=1)  # (23, )

        canonical_joints = pose[1:, :3, 3]  # (n_bone, 3)
        canonical_parent_joints = pose[self.parent_id[1:], :3, 3]  # (n_bone, 3)
        self.register_buffer('canonical_joints', torch.tensor(canonical_joints, dtype=torch.float32))
        self.register_buffer('canonical_parent_joints', torch.tensor(canonical_parent_joints, dtype=torch.float32))

        if self.origin_location == "center":
            # move origins to parts' center (self.origin_location == "center)
            pose = np.concatenate([pose[1:, :, :3],
                                   (pose[1:, :, 3:] +
                                    pose[self.parent_id[1:], :, 3:]) / 2], axis=-1)  # (23, 4, 4)
        elif self.origin_location == "center_fixed":
            pose = np.concatenate([pose[self.parent_id[1:], :, :3],
                                   (pose[1:, :, 3:] +
                                    pose[self.parent_id[1:], :, 3:]) / 2], axis=-1)  # (23, 4, 4)
        elif self.origin_location == "center+head":
            length = np.concatenate([length, np.ones(1, )])  # (24,)
            head_id = 15
            _pose = np.concatenate([pose[self.parent_id[1:], :, :3],
                                    (pose[1:, :, 3:] +
                                     pose[self.parent_id[1:], :, 3:]) / 2], axis=-1)  # (23, 4, 4)
            pose = np.concatenate([_pose, pose[head_id][None]])  # (24, 4, 4)

        self.register_buffer('canonical_bone_length', torch.tensor(length, dtype=torch.float32))
        self.register_buffer('canonical_pose', torch.tensor(pose, dtype=torch.float32))

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

    def to_local_and_canonical(self, points, pose_to_camera, bone_length):
        """transform points to local and canonical coordinate

        Args:
            points:
            pose_to_camera:
            bone_length: (B, n_bone, 1)

        Returns:

        """
        # to local coordinate
        R = pose_to_camera[:, :, :3, :3]  # (B, n_bone, 3, 3)
        inv_R = R.permute(0, 1, 3, 2)
        t = pose_to_camera[:, :, :3, 3:]  # (B, n_bone, 3, 1)
        local_points = torch.matmul(inv_R, points[:, None] - t)  # (B, n_bone, 3, n)

        # to canonical coordinate
        canonical_scale = (self.canonical_bone_length[:, None] / bone_length / self.coordinate_scale)[:, :, :, None]
        canonical_points = local_points * canonical_scale
        canonical_R = self.canonical_pose[:, :3, :3]  # (n_bone, 3, 3)
        canonical_t = self.canonical_pose[:, :3, 3:]  # (n_bone, 3, 1)
        canonical_points = torch.matmul(canonical_R, canonical_points) + canonical_t

        # reshape local
        bs, n_bone, _, n = local_points.shape
        local_points = local_points.reshape(bs, n_bone * 3, n)
        return local_points, canonical_points

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


class SSONARF(NARFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=False, parent=None, num_bone_param=None, view_dependent: bool = True):
        assert config.origin_location in ["center", "center_fixed"]
        self.tri_plane_based = False
        super(SSONARF, self).__init__(config, z_dim, num_bone, bone_length, parent, num_bone_param, view_dependent)
        hidden_size = self.hidden_size

        # selector
        hidden_dim_for_mask = 10
        self.selector = nn.Sequential(nn.Conv1d(3 * self.num_frequency_for_position * 2 * self.num_bone,
                                                hidden_dim_for_mask * self.num_bone, 1, groups=self.num_bone),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(hidden_dim_for_mask * self.num_bone, self.num_bone, 1,
                                                groups=self.num_bone),
                                      nn.Softmax(dim=1))

        if self.config.model_type == "dnarf":
            self.deformation_field = MLP((self.num_bone * 3 + 1) * self.num_frequency_for_position * 2, hidden_size,
                                         self.num_bone * 3, num_layers=8, skips=(4,))
            self.density_mlp = MLP(self.num_bone * 3 * self.num_frequency_for_position * 2, hidden_size, hidden_size,
                                   num_layers=8, skips=(4,))
        elif self.config.model_type == "tnarf":
            self.density_mlp = StyledMLP(self.num_bone * 3 * self.num_frequency_for_position * 2, hidden_size,
                                         hidden_size, style_dim=self.z_dim, num_layers=8)
        elif self.config.model_type == "narf":
            self.density_mlp = MLP(self.num_bone * 3 * self.num_frequency_for_position * 2, hidden_size,
                                   hidden_size, num_layers=8, skips=(4,))

        self.density_fc = StyledConv1d(self.hidden_size, 1, self.z2_dim)
        if view_dependent:
            self.mlp = StyledMLP(self.hidden_size + 3 * self.num_frequency_for_other * 2, self.hidden_size // 2,
                                 3, style_dim=self.z2_dim)
        else:
            self.mlp = StyledMLP(self.hidden_size, self.hidden_size // 2, 3, style_dim=self.z2_dim)

    @staticmethod
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

    def calc_density_and_color_from_camera_coord(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                 bone_length: torch.Tensor, z, z_rend, ray_direction):
        """compute density from positions in camera coordinate

        :param position:
        :param pose_to_camera:
        :param bone_length:
        :param z:
        :param z_rend:
        :return: density of input positions
        """
        local_points = self.to_local(position, pose_to_camera)

        in_cube_p = in_cube(local_points)  # (B, n_bone, n)
        density, color = self.backbone(local_points, in_cube_p, z, z_rend, bone_length, ray_direction)
        density *= in_cube_p.any(dim=1, keepdim=True)
        return density, color

    def backbone(self, p: torch.Tensor, position_validity: torch.Tensor, z: torch.Tensor,
                 z_rend: torch.Tensor, bone_length: torch.Tensor,
                 ray_direction: Optional[torch.Tensor] = None):
        """

        Args:
            p: position in local coordinate, (B, n_bone, 3, n)
            position_validity: bool tensor for validity of p, (B, n_bone, n)
            z: (B, dim)
            z_rend: (B, dim)
            bone_length: (B, n_bone)
            # mode: "weight_feature" or "weight_position"
            ray_direction: not None if color is view dependent
        Returns:

        """
        # don't support mip-nerf rendering
        assert isinstance(p, torch.Tensor)
        assert bone_length is not None
        # assert mode in ["weight_position", "weight_feature"]
        encoded_p = encode(p, self.num_frequency_for_position, self.num_bone)
        prob = self.selector(encoded_p)

        encoded_p = encoded_p * torch.repeat_interleave(prob, 3 * self.num_frequency_for_position * 2, dim=1)

        if self.config.model_type == "dnarf":
            expand_z = z[:, :, None].expand(-1, -1, p.shape[-1])
            dp = self.deformation_field(torch.cat([encoded_p, expand_z], dim=1))  # (B, num_bone * 3, n)
            p = p + dp
            encoded_p = encode(p, self.num_frequency_for_position, self.num_bone)

        if self.config.model_type == "tnarf":
            feature = self.density_mlp(encoded_p, z)
        else:
            feature = self.density_mlp(encoded_p)

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
