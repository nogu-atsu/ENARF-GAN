import sys
import warnings
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from NARF.models.activation import MyReLU
from NARF.models.model_utils import in_cube
from NARF.models.nerf_model import NeRF
from models.stylegan import StyledConv, EqualConv1d

sys.path.append("stylegan2-ada-pytorch")
import dnnlib

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

        # self.temporal_state = {
        #     "coarse_density": coarse_density,
        #     "coarse_T_i": T_i,
        #     "coarse_weights": weights,
        #     "coarse_depth": coarse_depth,
        #     "fine_depth": fine_depth,
        #     "fine_points": fine_points,
        #     "near_plane": near_plane,
        #     "far_plane": far_plane
        # }

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


class TriPlaneNeRF(StyleNeRF):
    def __init__(self, config, z_dim=256, num_bone=1, bone_length=True, parent=None, num_bone_param=None):
        super(NeRF, self).__init__()
        assert bone_length
        assert num_bone_param is not None
        assert hasattr(config, "origin_location")

        self.config = config
        hidden_size = config.hidden_size
        # use_world_pose = not config.no_world_pose
        # use_ray_direction = not config.no_ray_direction
        self.final_activation = config.final_activation
        self.origin_location = config.origin_location
        self.coordinate_scale = config.coordinate_scale
        self.mip_nerf_resolution = config.mip_nerf_resolution
        # self.mip_nerf = config.mip_nerf
        # assert self.final_activation in ["tanh", "l2", None]
        assert self.origin_location == "center"
        assert parent is not None
        # assert (self.mip_nerf_resolution is not None) == self.config.mip_nerf

        # dim = 3  # xyz
        # num_mlp_layers = 3
        # self.out_dim = config.out_dim if "out_dim" in self.config else 3
        self.parent_id = parent
        self.use_bone_length = bone_length
        # self.mask_before_PE = False
        # self.group_conv_first = config.group_conv_first

        self.mask_input = self.config.concat and self.config.mask_input
        self.selector_activation = self.config.selector_activation
        selector_tmp = self.config.selector_adaptive_tmp.start
        self.register_buffer("selector_tmp", torch.tensor(selector_tmp).float())

        self.density_activation = MyReLU.apply

        self.density_scale = config.density_scale

        # parameters for position encoding
        nffo = self.config.num_frequency_for_other if "num_frequency_for_other" in self.config else 4
        self.num_frequency_for_other = nffo

        self.hidden_size = hidden_size
        self.num_bone = num_bone - 1
        self.num_bone_param = num_bone_param if num_bone_param is not None else num_bone
        assert self.num_bone == self.num_bone_param
        self.z_dim = z_dim * 2  # TODO fix this

        self.fc_bone_length = torch.jit.script(
            StyledConv1d(self.num_frequency_for_other * 2 * self.num_bone_param,
                         self.z_dim, self.z_dim))
        self.tri_plane_gen = self.prepare_stylegan2()
        self.mlp = StyledMLP(32, 64, 4, style_dim=z_dim)

        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def prepare_stylegan2(self):
        G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=self.z_dim, w_dim=self.z_dim,
                                   mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
        G_kwargs.synthesis_kwargs.channel_base = 32768
        G_kwargs.synthesis_kwargs.channel_max = 512
        G_kwargs.mapping_kwargs.num_layers = 8
        G_kwargs.synthesis_kwargs.num_fp16_res = 0
        G_kwargs.synthesis_kwargs.conv_clamp = None
        g_common_kwargs = dict(c_dim=self.num_frequency_for_other * 2 * self.num_bone_param,
                               img_resolution=256, img_channels=(32 + self.num_bone) * 3)
        gen = dnnlib.util.construct_class_by_name(**G_kwargs, **g_common_kwargs)
        return gen

    def register_canonical_pose(self, pose: np.ndarray) -> None:
        """ register canonical pose.

        Args:
            pose: array of (24, 4, 4)

        Returns:

        """
        assert self.origin_location == "center"
        coordinate = pose[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parent_id[1:]], axis=1)  # (23, )

        # move origins to parts' center (self.origin_location == "center)
        pose = torch.cat([pose[1:, :, :3],
                          (pose[1:, :, 3:] +
                           pose[self.parent_id[1:], :, 3:]) / 2], dim=-1)  # (23, 4, 4)

        self.register_buffer('canonical_bone_length', torch.tensor(length, dtype=torch.float32))
        self.register_buffer('canonical_pose', torch.tensor(pose, dtype=torch.float32))

    @staticmethod
    def sample_feature(tri_plane_features: torch.tensor, position: torch.tensor):
        """sample tri-plane feature at a position

        :param tri_plane_features: (B, feat_dim * 3, size, size)
        :param position: [-1, 1] in meter, (B, 3, n)

        :return: feature: (B, 32, n)
        """
        batchsize, _, h, w = tri_plane_features.shape
        _, _, n = position.shape
        features = tri_plane_features.reshape(batchsize * 3, -1, h, w)
        position_2d = position[:, [0, 1, 1, 2, 2, 0]].reshape(batchsize * 3, 2, n)
        position_2d = position_2d.permute(0, 2, 1)[:, :, None]
        feature = F.grid_sample(features, position_2d)
        feature = feature.reshape(batchsize, 3, -1, n)
        feature = feature.sum(dim=1)  # (B, feat_dim, n)
        return feature

    @staticmethod
    def sample_point_feature(tri_plane_features: torch.tensor, position: torch.tensor, padding_value: float = 0):
        """sample tri-plane feature at a position

        :param tri_plane_features: (B, feat_dim * 3, size, size)
        :param position: [-1, 1] in meter, (B, num_bone, 3, n)
        :param padding_value: padding for F.grid_sample

        :return: feature: (B, num_bone, 32, n)
        """
        batchsize, _, h, w = tri_plane_features.shape
        _, n_bone, _, n = position.shape
        features = tri_plane_features.reshape(batchsize * 3, -1, h, w)
        position_2d = position[:, :, [0, 1, 1, 2, 2, 0]].reshape(batchsize, n_bone, 3, 2, n)
        position_2d = position_2d.permute(0, 2, 1, 4, 3).reshape(batchsize * 3, n_bone, n, 2)
        feature = F.grid_sample(features, position_2d)  # , mode="nearest")
        feature = feature.reshape(batchsize, 3, -1, n_bone, n)
        if padding_value is not 0:
            # if all elements of feature is 0, fill padding value
            feature = feature.masked_fill(feature.square().sum(dim=2, keepdim=True) < 1e5, padding_value)
        feature = feature.sum(dim=1)  # (B, feat_dim, n_bone, n)
        return feature

    def calc_weight(self, tri_plane_weights: torch.tensor, position: torch.tensor):
        bs, n_bone, _, n = position.shape
        position = position.reshape(bs * n_bone, 1, 3, n)
        weight = self.sample_point_feature(tri_plane_weights, position, padding_value=-1e8)  # (B * n_bone, 1, 1, n)
        weight = weight.reshape(bs, n_bone, n)
        weight = torch.softmax(weight, dim=1)
        return weight

    def calc_color_and_density(self, local_pos: torch.Tensor, canonical_pos: torch.Tensor, z: torch.Tensor,
                               z_rend: torch.Tensor, bone_length: torch.Tensor, mode: str):
        """
        forward func of ImplicitField
        :param local_pos: local coordinate, (B, n_bone, 3, n) (n = num_of_ray * points_on_ray)
        :param canonical_pos: canonical coordinate, (B, n_bone, 3, n) (n = num_of_ray * points_on_ray)
        :param z: b x dim
        :param z_rend: b x groups x 4 x 4
        :param bone_length: b x groups x 1
        :param mode: str
        :return: b x groups x 4 x n
        """

        density, color = self.backbone(canonical_pos, z, z_rend, bone_length, mode)
        in_cube_p = in_cube(local_pos)
        density *= in_cube_p.any(dim=1, keepdim=True)
        return density, color  # B x groups x 1 x n, B x groups x 3 x n

    def to_local_and_canonical(self, points, pose_to_camera, bone_length):
        # to local coordinate
        R = pose_to_camera[:, :, :3, :3]  # (B, n_bone, 3, 3)
        inv_R = R.permute(0, 1, 3, 2)
        t = pose_to_camera[:, :, :3, 3:]  # (B, n_bone, 3, 1)
        local_points = torch.matmul(inv_R, points[:, None] - t)  # (B, n_bone, 3, n*Nc)

        # to canonical coordinate
        canonical_scale = (bone_length / self.canonical_bone_length / self.coordinate_scale)[None, :, None, None]
        canonical_points = local_points * canonical_scale
        canonical_R = self.canonical_pose[:, :3, :3]  # (n_bone, 3, 3)
        canonical_t = self.canonical_pose[:, :3, 3:]  # (n_bone, 3, 1)
        canonical_points = torch.matmul(canonical_R, canonical_points) + canonical_t

        return local_points, canonical_points

    def coarse_to_fine_sample(self, image_coord: torch.tensor, pose_to_camera: torch.tensor,
                              inv_intrinsics: torch.tensor, z: torch.tensor, z_rend: torch.tensor,
                              bone_length: torch.tensor = None, near_plane: float = 0.3, far_plane: float = 5,
                              Nc: int = 64, Nf: int = 128, render_scale: float = 1
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        batchsize, _, _, n = image_coord.shape
        num_bone = self.num_bone
        with torch.no_grad():
            (depth_min, depth_max, ray_direction) = self.decide_frustrum_range(num_bone, image_coord, pose_to_camera,
                                                                               inv_intrinsics, near_plane,
                                                                               far_plane, return_camera_coord=True)
            depth_min = depth_min.squeeze(1)
            depth_max = depth_max.squeeze(1)
            start = depth_min * ray_direction  # (B, 3, n)
            end = depth_max * ray_direction  # (B, 3, n)
            # coarse ray sampling
            bins = torch.linspace(0, 1, Nc + 1, dtype=torch.float, device="cuda").reshape(1, 1, 1, Nc + 1)
            coarse_depth = (depth_min.unsqueeze(-1) * (1 - bins) +
                            depth_max.unsqueeze(-1) * bins)  # (B, 1, n, Nc + 1)

            coarse_points = start.unsqueeze(-1) * (1 - bins) + end.unsqueeze(-1) * bins  # (B, 3, n, (Nc+1))
            coarse_points = (coarse_points[:, :, :, 1:] + coarse_points[:, :, :, :-1]) / 2
            coarse_points = coarse_points.reshape(batchsize, 3, -1)

            local_coarse_points, canonical_coarse_points = self.to_local_and_canonical(coarse_points, pose_to_camera,
                                                                                       bone_length)
            # coarse density
            coarse_density = self.calc_color_and_density(local_coarse_points, canonical_coarse_points,
                                                         z, z_rend, bone_length, mode="weight_position")[
                0]  # B x groups x n*Nc

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
            weights = F.pad(weights, (1, 1, 0, 0))
            weights = (torch.maximum(weights[:, :-2], weights[:, 1:-1]) +
                       torch.maximum(weights[:, 1:-1], weights[:, 2:])) / 2 + 0.01
            bins = (torch.multinomial(weights,
                                      Nf, replacement=True).reshape(batchsize, 1, n, Nf).float() / Nc +
                    torch.cuda.FloatTensor(batchsize, 1, n, Nf).uniform_() / Nc)

            # sort points
            bins = torch.sort(bins, dim=-1)[0]
            fine_depth = depth_min.unsqueeze(3) * (1 - bins) + depth_max.unsqueeze(3) * bins  # (B, 1, n, Nf)

            fine_points = start.unsqueeze(3) * (1 - bins) + end.unsqueeze(3) * bins  # (B, 3, n, Nf)

            fine_points = fine_points.reshape(batchsize, 3, -1)

        # self.temporal_state = {
        #     "coarse_density": coarse_density,
        #     "coarse_T_i": T_i,
        #     "coarse_weights": weights,
        #     "coarse_depth": coarse_depth,
        #     "fine_depth": fine_depth,
        #     "fine_points": fine_points,
        #     "near_plane": near_plane,
        #     "far_plane": far_plane
        # }

        if pose_to_camera.requires_grad:
            raise NotImplementedError("Currently pose should not be differentiable")

        return (
            fine_depth,  # (B, 1, n, Nf)
            fine_points,  # (B, 3, n*Nf)
        )

    def backbone(self, p: torch.Tensor, z: torch.Tensor, z_rend: torch.Tensor, bone_length: torch.Tensor,
                 mode: str = "weight_feature"):
        """

        Args:
            p: position in canonical coordinate, (B, 1, 3, n)
            z: (B, dim)
            z_rend: (B, dim)
            bone_length: (B, n_bone)
            mode: "weight_feature" or "weight_position"
        Returns:

        """
        # don't support mip-nerf rendering
        assert isinstance(p, torch.Tensor)
        assert z is not None
        assert bone_length is not None
        assert mode in ["weight_position", "weight_feature"]

        # forward外
        # canonical coordinateに変換
        # -generate tri-plane feature conditioned on z and bone_length
        encoded_length = encode(bone_length, self.num_frequency_for_other, num_bone=self.num_bone_param)
        tri_plane_feature = self.gen(z, encoded_length[:, :, 0])  # (B, (32 + n_bone) * 3, h, w)

        # forward steps
        # mask valueを取得，weight計算
        # 1. 重みづけ平均で座標をmap
        # # feature 取得
        # 2. local 座標にmapする
        # # featureを取得して，weightでaverage

        bs, n_bone, _, n = p.shape
        weight = self.calc_weight(tri_plane_feature[:, 32 * 3:].reshape(bs * 23, 3, 256, 256), p)  # (bs, n_bone, n)
        weighted_position = (p * weight[:, :, None]).sum(dim=1)  # (bs, 3, n)
        # canonical position based
        if mode == "weight_position":
            feature = self.sample_feature(tri_plane_feature[:, :32 * 3], weighted_position)  # (B, 32, n)
            color_density = self.mlp(feature, z_rend)  # (B, 4, n)

        # concat position based
        elif mode == "weight_feature":
            feature = self.sample_point_feature(tri_plane_feature[:, :32 * 3], p)  # (B, 32, n_bone, n)
            feature = torch.sum(feature * weight[:, None], dim=2)  # (B, 32, n)
            color_density = self.mlp(feature, z_rend)  # (B, 4, n)
        else:
            raise ValueError()
        color, density = color_density[:, :3], color_density[:, 3:]
        return density, color

    def render(self, image_coord: torch.tensor, pose_to_camera: torch.tensor, inv_intrinsics: torch.tensor,
               z: torch.tensor, z_rend: torch.tensor, bone_length: torch.tensor,
               thres: float = 0.0, render_scale: float = 1, Nc: int = 64, Nf: int = 128,
               semantic_map: bool = False, return_intermediate: bool = False) -> (torch.tensor,) * 3:
        near_plane = 0.3
        # n <- number of sampled pixels
        # image_coord: B x groups x 3 x n
        # camera_extrinsics: B x 4 x 4
        # camera_intrinsics: 3 x 3

        batchsize, num_bone, _, n = image_coord.shape

        if self.origin_location == "center":
            pose_to_camera = torch.cat([pose_to_camera[:, 1:, :, :3],
                                        (pose_to_camera[:, 1:, :, 3:] +
                                         pose_to_camera[:, self.parent_id[1:], :, 3:]) / 2], dim=-1)

        if self.coordinate_scale != 1:
            pose_to_camera[:, :, :3, 3] *= self.coordinate_scale

        fine_depth, fine_points = self.coarse_to_fine_sample(image_coord, pose_to_camera,
                                                             inv_intrinsics,
                                                             z=z, z_rend=z_rend,
                                                             bone_length=bone_length,
                                                             near_plane=near_plane, Nc=Nc, Nf=Nf,
                                                             render_scale=render_scale)
        # fine density & color # B x groups x 1 x n*(Nc+Nf), B x groups x 3 x n*(Nc+Nf)
        if semantic_map and self.config.mask_input:
            self.save_mask = True

        local_fine_points, canonical_fine_points = self.to_local_and_canonical(fine_points, pose_to_camera, bone_length)

        fine_density, fine_color = self.calc_color_and_density(local_fine_points, canonical_fine_points, z, z_rend,
                                                               bone_length, mode="weight_position")
        Np = fine_depth.shape[-1]  # Nf

        if return_intermediate:
            intermediate_output = (fine_points, fine_density)

        if semantic_map and self.config.mask_input:
            self.save_mask = False

        # semantic map
        if semantic_map:
            assert False, "semantic map rendering will be implemented later"
            bone_idx = torch.arange(num_bone).cuda()
            seg_color = torch.stack([bone_idx // 9, (bone_idx // 3) % 3, bone_idx % 3], dim=1) - 1  # num_bone x 3
            seg_color[::2] = seg_color.flip(dims=(0,))[1 - num_bone % 2::2]  # num_bone x 3
            fine_color = seg_color[self.mask_prob.reshape(-1)]
            fine_color = fine_color.reshape(batchsize, 1, -1, 3).permute(0, 1, 3, 2)
            fine_color = fine_color.reshape(batchsize, 1, 3, n, -1)[:, :, :, :, :Np - 1]
        else:
            fine_color = fine_color.reshape(batchsize, 1, self.out_dim, n, -1)[:, :, :, :, :Np - 1]

        fine_density = fine_density.reshape(batchsize, 1, 1, n, -1)[:, :, :, :, :Np - 1]

        sum_fine_density = fine_density

        # if thres > 0:
        #     # density = inf if density exceeds thres
        #     sum_fine_density = (sum_fine_density > thres) * 100000

        delta = fine_depth[:, :, :, :, 1:] - fine_depth[:, :, :, :, :-1]  # B x 1 x 1 x n x Np-1
        sum_density_delta: torch.Tensor = sum_fine_density * delta * render_scale  # B x 1 x 1 x n x Np-1

        T_i = torch.exp(-(torch.cumsum(sum_density_delta, dim=4) - sum_density_delta))
        weights = T_i * (1 - torch.exp(-sum_density_delta))  # B x 1 x 1 x n x Nc+Nf-1

        if not hasattr(self, "buffers_tensors"):
            self.buffers_tensors = {}
        self.buffers_tensors["fine_weights"] = weights
        self.buffers_tensors["fine_depth"] = fine_depth

        # self.temporal_state["fine_density"] = fine_density
        # self.temporal_state["fine_T_i"] = T_i
        # self.temporal_state["fine_cum_weight"] = torch.cumsum(weights, dim=-1)

        fine_depth = fine_depth.reshape(batchsize, 1, 1, n, -1)[:, :, :, :, :-1]

        rendered_color = torch.sum(weights * fine_color, dim=4).squeeze(1)  # B x 3 x n
        rendered_mask = torch.sum(weights, dim=4).reshape(batchsize, n)  # B x n
        rendered_disparity = torch.sum(weights * 1 / fine_depth, dim=4).reshape(batchsize, n)  # B x n

        if return_intermediate:
            return rendered_color, rendered_mask, rendered_disparity, intermediate_output

        return rendered_color, rendered_mask, rendered_disparity

    def forward(self, batchsize, sampled_img_coord, pose_to_camera, inv_intrinsics, z, z_rend,
                bone_length, render_scale=1, Nc=64, Nf=128,
                return_intermediate=False):
        """
        rendering function for sampled rays
        :param batchsize:
        :param sampled_img_coord: sampled image coordinate
        :param pose_to_camera:
        :param inv_intrinsics:
        :param z:
        :param world_pose:
        :param bone_length:
        :param thres:
        :param render_scale:
        :param Nc:
        :param Nf:
        :param return_intermediate:
        :return: color and mask value for sampled rays
        """

        nerf_output = self.render(sampled_img_coord,
                                  pose_to_camera,
                                  inv_intrinsics,
                                  z=z,
                                  z_rend=z_rend,
                                  bone_length=bone_length,
                                  Nc=Nc,
                                  Nf=Nf,
                                  render_scale=render_scale,
                                  return_intermediate=return_intermediate)
        if return_intermediate:
            merged_color, merged_mask, _, intermediate_output = nerf_output
            return merged_color, merged_mask, intermediate_output

        merged_color, merged_mask, _ = nerf_output
        return merged_color, merged_mask
