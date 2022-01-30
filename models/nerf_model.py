import warnings
from typing import Optional, Union, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from NARF.models.activation import MyReLU
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


class StyledMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, style_dim=512, num_layers=3):
        super(StyledMLP, self).__init__()
        # TODO use StyledConv
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


class StyleNeRF(NeRF):
    def __init__(self, config, z_dim=256, num_bone=1, bone_length=True, parent=None, num_bone_param=None):
        super(NeRF, self).__init__()
        self.config = config
        hidden_size = config.hidden_size
        use_world_pose = not config.no_world_pose
        use_ray_direction = not config.no_ray_direction
        self.final_activation = config.final_activation
        self.origin_location = config.origin_location if hasattr(config, "origin_location") else "root"
        self.coordinate_scale = config.coordinate_scale
        self.mip_nerf_resolution = config.mip_nerf_resolution
        self.mip_nerf = config.mip_nerf
        assert self.final_activation in ["tanh", "l2", None]
        assert self.origin_location in ["root", "center"]
        assert self.origin_location == "root" or parent is not None
        assert (self.mip_nerf_resolution is not None) == self.config.mip_nerf

        dim = 3  # xyz
        num_mlp_layers = 3
        self.out_dim = config.out_dim if "out_dim" in self.config else 3
        self.parent_id = parent
        self.use_bone_length = bone_length
        self.mask_before_PE = False
        self.group_conv_first = config.group_conv_first

        self.mask_input = self.config.concat and self.config.mask_input
        self.selector_activation = self.config.selector_activation
        selector_tmp = self.config.selector_adaptive_tmp.start
        self.register_buffer("selector_tmp", torch.tensor(selector_tmp).float())

        self.groups = 1

        self.density_activation = MyReLU.apply

        self.density_scale = config.density_scale

        # parameters for position encoding
        nffp = self.config.num_frequency_for_position if "num_frequency_for_position" in self.config else 10
        nffo = self.config.num_frequency_for_other if "num_frequency_for_other" in self.config else 4

        self.num_frequency_for_position = nffp
        self.num_frequency_for_other = nffo

        self.hidden_size = hidden_size
        self.num_bone = num_bone - 1 if self.origin_location == "center" else num_bone
        self.num_bone_param = num_bone_param if num_bone_param is not None else num_bone
        self.z_dim = z_dim * 2  # TODO fix this

        if self.group_conv_first:
            fc_p_groups = 5
            fc_p_out_dim = hidden_size // fc_p_groups * fc_p_groups
        else:
            fc_p_groups = 1
            fc_p_out_dim = hidden_size

        self.fc_p = torch.jit.script(StyledConv1d(dim * self.num_frequency_for_position * 2 * self.num_bone,
                                                  fc_p_out_dim, self.z_dim, groups=fc_p_groups))
        if bone_length:
            assert not self.config.weighted_average
            self.fc_bone_length = torch.jit.script(
                StyledConv1d(self.num_frequency_for_other * 2 * self.num_bone_param,
                             fc_p_out_dim, self.z_dim))

        self.use_world_pose = use_world_pose
        self.use_ray_direction = use_ray_direction

        assert not use_world_pose

        if use_ray_direction:
            self.fc_d = StyledConv1d(dim * self.num_frequency_for_other * 2 * self.num_bone,
                                     hidden_size // 2, self.z_dim)

        if self.mask_input:
            print("mask input")
            hidden_dim_for_mask = 10
            self.mask_linear_p = torch.jit.script(
                StyledConv1d(dim * self.num_frequency_for_position * 2 * self.num_bone,
                             hidden_dim_for_mask * self.num_bone, self.z_dim, groups=self.num_bone))
            self.mask_linear_l = torch.jit.script(
                StyledConv1d(self.num_frequency_for_other * 2 * self.num_bone_param,
                             hidden_dim_for_mask * self.num_bone, self.z_dim))
            self.mask_linear = torch.jit.script(
                StyledConv1d(hidden_dim_for_mask * self.num_bone, self.num_bone, self.z_dim))

        self.density_mlp = torch.jit.script(StyledMLP(fc_p_out_dim, hidden_size, hidden_size, style_dim=self.z_dim,
                                                      num_layers=num_mlp_layers))
        # self.density_fc = torch.jit.script(StyledConv1d(hidden_size, 1, self.z_dim))
        self.density_fc = EqualConv1d(hidden_size, 1, 1)
        self.density_fc.weight.data[:] = 0

        # zero

        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    @property
    def memory_cost(self):
        raise NotImplementedError()
        m = 0
        for layer in self.children():
            if isinstance(layer, (NormalizedConv1d, EqualLinear, EqualConv1d, MLP)):
                m += layer.memory_cost
        return m

    @property
    def flops(self):
        raise NotImplementedError()
        fl = 0
        for layer in self.children():
            if isinstance(layer, (NormalizedConv1d, EqualLinear, EqualConv1d, MLP)):
                fl += layer.flops

        if self.z_dim > 0:
            fl += self.hidden_size * 2
        if self.use_bone_length:
            fl += self.hidden_size
        fl += self.hidden_size * 2
        return fl

    def get_mu_sigma(self, camera_origin: torch.tensor, ray_direction: torch.tensor, depth: torch.tensor,
                     inv_intrinsics: torch.tensor):
        """compute mu and sigma for IPE in mip-NeRF

        Args:
            camera_origin: (B, n_bone, 3, n_ray)
            ray_direction: (B, n_bone, 3, n_ray)
            depth: (B, 1, 1, n_ray, Np)

        Returns: mu, sigma
        """
        t_mu = (depth[:, :, :, :, :-1] + depth[:, :, :, :, 1:]) / 2  # B x 1 x 1 x n x Nc
        t_delta = (depth[:, :, :, :, 1:] - depth[:, :, :, :, :-1]) / 2

        mu_t = t_mu + 2 * t_mu * t_delta ** 2 / (3 * t_mu ** 2 + t_delta ** 2)
        sigma_t2 = t_delta ** 2 / 3 - 4 / 15 * t_delta ** 4 * (12 * t_mu ** 2 - t_delta ** 2) / (
                3 * t_mu ** 2 + t_delta ** 2) ** 2
        r_dot = 2 / 3 ** 0.5 * inv_intrinsics[:, 0, 2] / self.mip_nerf_resolution  # (B, )
        sigma_r2 = r_dot[:, None, None, None, None] ** 2 * (
                t_mu ** 2 / 4 + 5 / 12 * t_delta ** 2 - 4 / 15 * t_delta ** 4 / (
                3 * t_mu ** 2 + t_delta ** 2))

        mu = camera_origin.unsqueeze(4) + mu_t * ray_direction.unsqueeze(4)  # (B, n_bone, 3, n, Nc)
        diag_sigma = (sigma_t2 * ray_direction.unsqueeze(4) ** 2 +
                      sigma_r2 * (1 - F.normalize(ray_direction, dim=2).unsqueeze(4) ** 2))
        return mu, diag_sigma

    def coarse_to_fine_sample(self, image_coord: torch.tensor, pose_to_camera: torch.tensor,
                              inv_intrinsics: torch.tensor, z: torch.tensor = None, world_pose: torch.tensor = None,
                              bone_length: torch.tensor = None, near_plane: float = 0.3, far_plane: float = 5,
                              Nc: int = 64, Nf: int = 128, render_scale: float = 1) -> (torch.tensor,) * 3:
        batchsize, _, _, n = image_coord.shape
        num_bone = 1 if self.config.concat_pose else self.num_bone  # PoseConditionalNeRF or other
        with torch.no_grad():
            (camera_origin, depth_min,
             depth_max, ray_direction) = self.decide_frustrum_range(num_bone, image_coord, pose_to_camera,
                                                                    inv_intrinsics, near_plane,
                                                                    far_plane)
            camera_origin = camera_origin.reshape(batchsize, num_bone, 3, n)

            # unit ray direction
            unit_ray_direction = F.normalize(ray_direction, dim=1)  # B*num_bone x 3 x n
            unit_ray_direction = unit_ray_direction.reshape(batchsize, num_bone * 3, n)
            ray_direction = ray_direction.reshape(batchsize, num_bone, 3, n)

            start = camera_origin + depth_min * ray_direction  # B x num_bone x 3 x n
            end = camera_origin + depth_max * ray_direction  # B x num_bone x 3 x n
            if self.mip_nerf:
                # coarse ray sampling
                bins = torch.linspace(0, 1, Nc + 1, dtype=torch.float, device="cuda").reshape(1, 1, 1, 1, Nc + 1)
                coarse_depth = (depth_min.unsqueeze(4) * (1 - bins) +
                                depth_max.unsqueeze(4) * bins)  # B x 1 x 1 x n x (Nc + 1)

                coarse_points, coarse_diag_sigma = self.get_mu_sigma(camera_origin, ray_direction, coarse_depth,
                                                                     inv_intrinsics)

                coarse_points = coarse_points.reshape(batchsize, num_bone * 3, -1)
                coarse_diag_sigma = coarse_diag_sigma.reshape(batchsize, num_bone * 3, -1)
                coarse_points = [coarse_points, coarse_diag_sigma]
                # coarse density
                coarse_density = self.calc_color_and_density(coarse_points, z, world_pose, bone_length,
                                                             unit_ray_direction, )[0]  # B x groups x n*Nc
            else:
                # coarse ray sampling
                bins = torch.arange(Nc, dtype=torch.float, device="cuda").reshape(1, 1, 1, 1, Nc) / Nc
                coarse_points = start.unsqueeze(4) * (1 - bins) + end.unsqueeze(4) * bins  # B x num_bone x 3 x n x Nc
                coarse_depth = (depth_min.unsqueeze(4) * (1 - bins) +
                                depth_max.unsqueeze(4) * bins)  # B x 1 x 1 x n x Nc

                # coarse density
                coarse_density = self.calc_color_and_density(coarse_points.reshape(batchsize, num_bone * 3, n * Nc),
                                                             z, world_pose,
                                                             bone_length,
                                                             unit_ray_direction)[0]  # B x groups x n*Nc

            if self.groups > 1:
                # alpha blending
                coarse_density, _ = self.sum_density(coarse_density)

            Np = coarse_depth.shape[-1]  # Nc or Nc + 1
            # calculate weight for fine sampling
            coarse_density = coarse_density.reshape(batchsize, 1, 1, n, Nc)[:, :, :, :, :Np - 1]
            # # delta = distance between adjacent samples
            delta = coarse_depth[:, :, :, :, 1:] - coarse_depth[:, :, :, :, :-1]  # B x 1 x 1 x n x Np - 1

            density_delta = coarse_density * delta * render_scale
            T_i = torch.exp(-(torch.cumsum(density_delta, dim=4) - density_delta))
            weights = T_i * (1 - torch.exp(-density_delta))  # B x 1 x 1 x n x Np-1
            weights = weights.reshape(batchsize * n, Np - 1)
            # fine ray sampling
            if self.mip_nerf:
                weights = F.pad(weights, (1, 1, 0, 0))
                weights = (torch.maximum(weights[:, :-2], weights[:, 1:-1]) +
                           torch.maximum(weights[:, 1:-1], weights[:, 2:])) / 2 + 0.01
                bins = (torch.multinomial(weights,
                                          Nf + 1, replacement=True).reshape(batchsize, 1, 1, n, Nf + 1).float() / Nc +
                        torch.cuda.FloatTensor(batchsize, 1, 1, n, Nf + 1).uniform_() / Nc)
            else:
                bins = (torch.multinomial(torch.clamp_min(weights, 1e-8),
                                          Nf, replacement=True).reshape(batchsize, 1, 1, n, Nf).float() / Nc -
                        torch.cuda.FloatTensor(batchsize, 1, 1, n, Nf).uniform_() / Nc)
            bins = torch.sort(bins, dim=-1)[0]
            fine_depth = (depth_min.unsqueeze(4) * (1 - bins) +
                          depth_max.unsqueeze(4) * bins)  # B x 1 x 1 x n x (Nf or Nf + 1)

            # sort points
            if self.mip_nerf:
                fine_points, fine_diag_sigma = self.get_mu_sigma(camera_origin, ray_direction, fine_depth,
                                                                 inv_intrinsics)
                fine_points = fine_points.reshape(batchsize, num_bone * 3, -1)
                fine_diag_sigma = fine_diag_sigma.reshape(batchsize, num_bone * 3, -1)
            else:
                fine_points = start.unsqueeze(4) * (1 - bins) + end.unsqueeze(4) * bins  # B x num_bone x 3 x n x Nf
                fine_points = torch.cat([coarse_points, fine_points], dim=4)
                fine_depth = torch.cat([coarse_depth, fine_depth], dim=4)
                arg = torch.argsort(fine_depth, dim=4)

                fine_points = torch.gather(fine_points, dim=4,
                                           index=arg.repeat(1, num_bone, 3, 1, 1))  # B x num_bone x 3 x n x Nc+Nf
                fine_depth = torch.gather(fine_depth, dim=4, index=arg)  # B x 1 x 1 x n x Nc+Nf

                fine_points = fine_points.reshape(batchsize, num_bone * 3, -1)

        self.temporal_state = {
            "coarse_density": coarse_density,
            "coarse_T_i": T_i,
            "coarse_weights": weights,
            "coarse_depth": coarse_depth,
            "fine_depth": fine_depth,
            "fine_points": fine_points,
            "near_plane": near_plane,
            "far_plane": far_plane
        }

        if pose_to_camera.requires_grad:
            R = pose_to_camera[:, :, :3, :3]
            t = pose_to_camera[:, :, :3, 3:]

            with torch.no_grad():
                fine_points = fine_points.reshape(batchsize, num_bone, 3, -1)
                fine_points = torch.matmul(R, fine_points) + t
            fine_points = torch.matmul(R.permute(0, 1, 3, 2), fine_points - t).reshape(batchsize, num_bone * 3, -1)

        if self.mip_nerf:
            fine_points = [fine_points, fine_diag_sigma]

        return (
            fine_depth,  # B x 1 x 1 x n x Nc+Nf
            fine_points,  # B x num_bone*3 x n*Nc+Nf
            unit_ray_direction  # B x num_bone*3 x n
        )

    def backbone_(self, p, z=None, j=None, bone_length=None, ray_direction=None):
        assert isinstance(p, list) == self.mip_nerf

        act = nn.LeakyReLU(0.2, inplace=True)

        def clac_p_and_length_feature(p, bone_length, z):
            if bone_length is not None:
                encoded_length = encode(bone_length, self.num_frequency_for_other, num_bone=self.num_bone_param)
            encoded_p = encode(p, self.num_frequency_for_position, num_bone=self.num_bone)

            _mask_prob = None
            if self.mask_input:
                net = self.mask_linear_p(encoded_p, z)
                if bone_length is not None:
                    net = net + self.mask_linear_l(encoded_length, z)
                input_mask = self.mask_linear(act(net), z)  # B x num_bone x n

                if self.selector_activation == "softmax":
                    _mask_prob = torch.softmax(input_mask / self.selector_tmp, dim=1)  # B x num_bone x n
                elif self.selector_activation == "sigmoid":
                    _mask_prob = torch.sigmoid(input_mask)  # B x num_bone x n
                else:
                    raise ValueError()
                if self.config.use_scale_factor:
                    scale_factor = self.num_bone ** 0.5 / torch.norm(_mask_prob, dim=1, keepdim=True)
                    _mask_prob = _mask_prob * scale_factor
                # if self.save_mask:  # save mask for segmentation rendering
                #     self.mask_prob = _mask_prob.argmax(dim=1).data.cpu().numpy()  # B x n

                encoded_p = self.apply_mask(None, encoded_p, _mask_prob,
                                            self.num_frequency_for_position)  # mask position
            batchsize = encoded_p.shape[0]
            encoded_p = encoded_p.reshape(batchsize, self.num_bone, 2, self.num_frequency_for_position * 3, -1)
            encoded_p = encoded_p.permute(0, 3, 1, 2, 4)
            encoded_p = encoded_p.reshape(batchsize, self.num_bone * self.num_frequency_for_position * 3 * 2, -1)

            net = self.fc_p(encoded_p, z)

            if bone_length is not None:
                net_bone_length = self.fc_bone_length(encoded_length, z)
                net = net + net_bone_length

            return net, _mask_prob

        net, mask_prob = clac_p_and_length_feature(p, bone_length, z)

        if j is not None and self.use_world_pose:
            assert False, "don't use world pose"

        batchsize, _, n = net.shape
        # ray direction
        if ray_direction is not None and self.use_ray_direction:
            warnings.warn("Using ray direction will not be supported in the future.")
            assert n % ray_direction.shape[2] == 0

            def calc_ray_feature(ray_direction, z):
                ray_direction = ray_direction.unsqueeze(3).repeat(1, 1, 1, n // ray_direction.shape[2])
                ray_direction = ray_direction.reshape(batchsize, -1, n)
                encoded_d = encode(ray_direction, self.num_frequency_for_other, num_bone=self.num_bone)

                if self.mask_input:
                    encoded_d = self.apply_mask(ray_direction, encoded_d, mask_prob,
                                                self.num_frequency_for_other)

                net_d = self.fc_d(encoded_d, z)
                return net_d

            net_d = calc_ray_feature(ray_direction, z)

        else:
            net_d = 0

        net = self.density_mlp(net, z)
        # density = self.density_fc(net, z)  # B x 1 x n
        density = self.density_fc(net)  # B x 1 x n
        density = self.density_activation(density)
        feature = net + net_d
        feature = act(feature)
        if self.final_activation == "tanh":
            feature = F.tanh(feature)
        elif self.final_activation == "l2":
            feature = F.normalize(feature, dim=2)
        return density, feature

    def nerf_path(self, ):
        raise NotImplementedError()
