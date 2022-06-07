import sys
import warnings
from typing import Union, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from NARF.models.activation import MyReLU
from NARF.models.nerf_model import NeRF
from models.nerf_model_base import NARFBase
from models.nerf_utils import StyledConv1d, encode, positional_encoding, in_cube
from models.stylegan import EqualConv1d
from cuda_extension.triplane_sampler import triplane_sampler

from stylegan2_ada_pytorch import dnnlib


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
        assert self.origin_location in ["root", "center", "center_fixed", "center+head"]
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
        self.num_bone = num_bone - 1 if self.origin_location in ["center", "center_fixed"] else num_bone
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

    # @property
    # def memory_cost(self):
    #     raise NotImplementedError()
    #     m = 0
    #     for layer in self.children():
    #         if isinstance(layer, (NormalizedConv1d, EqualLinear, EqualConv1d, MLP)):
    #             m += layer.memory_cost
    #     return m
    #
    # @property
    # def flops(self):
    #     raise NotImplementedError()
    #     fl = 0
    #     for layer in self.children():
    #         if isinstance(layer, (NormalizedConv1d, EqualLinear, EqualConv1d, MLP)):
    #             fl += layer.flops
    #
    #     if self.z_dim > 0:
    #         fl += self.hidden_size * 2
    #     if self.use_bone_length:
    #         fl += self.hidden_size
    #     fl += self.hidden_size * 2
    #     return fl

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
            feature = torch.tanh(feature)
        elif self.final_activation == "l2":
            feature = F.normalize(feature, dim=2)
        return density, feature

    def nerf_path(self, ):
        raise NotImplementedError()

    def transform_pose(self, pose_to_camera, bone_length):
        if self.origin_location == "center":
            pose_to_camera = torch.cat([pose_to_camera[:, 1:, :, :3],
                                        (pose_to_camera[:, 1:, :, 3:] +
                                         pose_to_camera[:, self.parent_id[1:], :, 3:]) / 2], dim=-1)
        elif self.origin_location == "center_fixed":
            pose_to_camera = torch.cat([pose_to_camera[:, self.parent_id[1:], :, :3],
                                        (pose_to_camera[:, 1:, :, 3:] +
                                         pose_to_camera[:, self.parent_id[1:], :, 3:]) / 2], dim=-1)

        elif self.origin_location == "center+head":
            bone_length = torch.cat([bone_length, torch.ones(bone_length.shape[0], 1, 1, device=bone_length.device)],
                                    dim=1)  # (B, 24)
            head_id = 15
            _pose_to_camera = torch.cat([pose_to_camera[:, self.parent_id[1:], :, :3],
                                         (pose_to_camera[:, 1:, :, 3:] +
                                          pose_to_camera[:, self.parent_id[1:], :, 3:]) / 2],
                                        dim=-1)  # (B, 23, 4, 4)
            pose_to_camera = torch.cat([_pose_to_camera, pose_to_camera[:, head_id][:, None]], dim=1)  # (B, 24, 4, 4)
        return pose_to_camera, bone_length

    def render_mesh(self, pose_to_camera, intrinsics, z, bone_length, voxel_size=0.003,
                    mesh_th=15):

        import mcubes
        from pytorch3d.renderer import (
            look_at_view_transform,
            FoVPerspectiveCameras,
            PointLights,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader,
            HardPhongShader,
            Textures,
        )
        from pytorch3d.structures import Meshes

        assert z is None or z.shape[0] == 1
        assert bone_length is None or bone_length.shape[0] == 1
        ray_batchsize = self.config.render_bs if hasattr(self.config, "render_bs") else 262144
        device = pose_to_camera.device
        cube_size = int(1 / voxel_size)

        center = pose_to_camera[:, 0, :3, 3:].clone()  # (1, 3, 1)

        bins = torch.arange(-cube_size, cube_size + 1) / cube_size
        p = (torch.stack(torch.meshgrid(bins, bins, bins)).reshape(1, 3, -1) + center.cpu()) * self.coordinate_scale

        pose_to_camera, bone_length = self.transform_pose(pose_to_camera, bone_length)

        if self.coordinate_scale != 1:
            pose_to_camera[:, :, :3, 3] *= self.coordinate_scale

        density = []
        for i in tqdm(range(0, p.shape[-1], ray_batchsize)):
            rot = pose_to_camera[:, :, :3, :3].permute(0, 1, 3, 2)
            trans = pose_to_camera[:, :, :3, 3:]
            p_i = p[:, :, i:i + ray_batchsize].cuda()  # (1, 3, ray_bs)
            local_p = torch.matmul(rot, (p_i[:, None] - trans))  # (1, n_bone, 3, ray_bs)
            local_p = local_p.reshape(1, self.num_bone * 3, -1)
            _density = self.calc_color_and_density(local_p, z, None, bone_length, None)[0]  # (1, 1, n)
            density.append(_density)
        density = torch.cat(density, dim=-1)
        density = density.reshape(cube_size * 2 + 1, cube_size * 2 + 1, cube_size * 2 + 1).cpu().numpy()

        vertices, triangles = mcubes.marching_cubes(density, mesh_th)
        vertices = (vertices - cube_size) * voxel_size  # (V, 3)
        vertices = torch.tensor(vertices, device=device).float() + center[:, :, 0]
        triangles = torch.tensor(triangles.astype("int64")).to(device)

        verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb)
        meshes = Meshes(verts=[vertices], faces=[triangles], textures=textures)

        cameras = FoVPerspectiveCameras(device=device, fov=30)  # , K=intrinsics)
        lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )
        images = renderer(meshes)
        images = images[0, :, :, :3]
        images = (images.cpu().numpy() * 255).astype("uint8")

        return images, meshes


class TriPlaneNARF(NARFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=True, parent=None, num_bone_param=None, view_dependent: bool = False):
        super(TriPlaneNARF, self).__init__()
        assert bone_length
        assert num_bone_param is not None
        assert hasattr(config, "origin_location")

        self.tri_plane_based = True
        self.config = config
        hidden_size = config.hidden_size
        # use_world_pose = not config.no_world_pose
        # use_ray_direction = not config.no_ray_direction
        # self.final_activation = config.final_activation
        self.origin_location = config.origin_location
        self.coordinate_scale = config.coordinate_scale
        # self.mip_nerf_resolution = config.mip_nerf_resolution
        # self.mip_nerf = config.mip_nerf
        # assert self.final_activation in ["tanh", "l2", None]
        assert self.origin_location in ["center", "center_fixed", "center+head"]
        assert parent is not None
        # assert (self.mip_nerf_resolution is not None) == self.config.mip_nerf

        # self.out_dim = config.out_dim if "out_dim" in self.config else 3
        self.parent_id = parent
        self.use_bone_length = bone_length

        # self.mask_input = self.config.concat and self.config.mask_input
        # self.selector_activation = self.config.selector_activation
        # selector_tmp = self.config.selector_adaptive_tmp.start
        # self.register_buffer("selector_tmp", torch.tensor(selector_tmp).float())

        self.density_activation = MyReLU.apply

        # self.density_scale = config.density_scale

        # parameters for position encoding
        nffp = self.config.num_frequency_for_position if "num_frequency_for_position" in self.config else 10
        nffo = self.config.num_frequency_for_other if "num_frequency_for_other" in self.config else 4
        self.num_frequency_for_position = nffp
        self.num_frequency_for_other = nffo

        self.hidden_size = hidden_size
        self.num_bone = num_bone - 1 if self.origin_location in ["center", "center_fixed"] else num_bone
        self.num_bone_param = num_bone_param if num_bone_param is not None else num_bone
        assert self.num_bone == self.num_bone_param
        if type(z_dim) == list:
            self.z_dim = z_dim[0]
            self.z2_dim = z_dim[1]
        else:
            self.z_dim = z_dim
            self.z2_dim = z_dim
        self.w_dim = 512
        self.feat_dim = 32

        # self.fc_bone_length = torch.jit.script(
        #     StyledConv1d(self.num_frequency_for_other * 2 * self.num_bone_param,
        #                  self.z_dim, self.z_dim))

        self.no_selector = self.config.no_selector
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

        self.view_dependent = view_dependent
        if view_dependent:
            self.density_fc = StyledConv1d(32, 1, self.z2_dim)
            self.mlp = StyledMLP(32 + 3 * nffo * 2, 64, 3, style_dim=self.z2_dim)
        else:
            print("not view dependent")
            self.mlp = StyledMLP(32, 64, 4, style_dim=self.z2_dim)

        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

        self.temporal_state = {}

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

    def sample_feature(self, tri_plane_features: torch.tensor, position: torch.tensor, reduction: str = "sum",
                       batch_idx: Optional[torch.Tensor] = None):
        """sample tri-plane feature at a position

        :param tri_plane_features: (B, ? * 3, h, w)
        :param position: [-1, 1] in meter, (B, 3, n)
        :param reduction: "prod" or "sum"
        :param batch_idx: index of data in minibatch

        :return: feature: (B, 32, n)
        """
        batchsize, _, h, w = tri_plane_features.shape
        assert batchsize == 1 or batch_idx is None
        _, _, n = position.shape
        if batchsize == 1 and reduction == "sum":
            print("called")
            position_2d = position.permute(0, 2, 1).contiguous()[:, :, None, :]
            feature = triplane_sampler(tri_plane_features, position_2d)[:, :, :, 0]
        else:
            features = tri_plane_features.reshape(batchsize * 3, -1, h, w)
            # produce 2D coordinate for each tri-plane
            position_2d = position[:, [0, 1, 1, 2, 2, 0]].reshape(batchsize * 3, 2, n)
            position_2d = position_2d.permute(0, 2, 1)[:, :, None]  # (B * 3, n, 1, 2)

            # if batch_idx is not None, place tri-planes side by side to form a single tri-plane (quite tricky)
            if batch_idx is not None:  # transform x coordinate
                actual_batchsize = w // (h + 1)
                scale = 1 / (actual_batchsize * (1 + 1 / h))
                position_2d[:, :, :, 0] = (position_2d[:, :, :, 0] * scale +
                                           batch_idx[None, :, None] * (2 / actual_batchsize) + (scale - 1))

            feature = F.grid_sample(features, position_2d, align_corners=False)
            # feature = torch.cudnn_grid_sampler(features, position_2d)
            feature = feature.reshape(batchsize, 3, -1, n)
            if reduction == "sum":
                feature = feature.sum(dim=1)  # (B, feat_dim, n)
            elif reduction == "prod":
                if self.config.clamp_mask:
                    feature = (feature.data.clamp(-2, 5) - feature.data) + feature
                feature = torch.sigmoid(feature).prod(dim=1)
            else:
                raise ValueError()
        return feature

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
            value = self.sample_feature(feature_padded, valid_positions,
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
        bs, n_bone, _, n = position.shape
        if self.no_selector:
            weight = torch.ones(bs, n_bone, n, device=position.device) / n_bone

        elif hasattr(self, "selector"):  # use selector
            position = position.reshape(bs, n_bone * 3, n)
            encoded_p = encode(position, self.num_frequency_for_position, self.num_bone)
            h = self.selector(encoded_p)
            weight = torch.softmax(h, dim=1)  # (B, n_bone, n)
        else:  # tri-plane based
            position = position.reshape(bs * n_bone, 3, n)

            # default mode is prod
            if mode == "prod":
                # sample prob from tri-planes and compute product
                weight = self.sample_feature(tri_plane_weights, position, reduction="prod")  # (B * n_bone, 1, n)
                weight = weight.reshape(bs, n_bone, n)
            elif mode == "sum":  # sum and softmax
                weight = self.sample_feature(tri_plane_weights, position)  # (B * n_bone, 1, n)
                weight = weight.reshape(bs, n_bone, n)

                # # wight for invalid point is 0
                weight = weight - ~position_validity * 1e4
                weight = torch.softmax(weight, dim=1)

            else:
                weight = torch.ones(bs, n_bone, n, device=position.device) / n_bone

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
            feature = self.sample_weighted_feature_v2(tri_plane_feature[:, :32 * 3], masked_position,
                                                      weight,
                                                      position_validity)  # (B, 32, n)
        # canonical position based
        elif mode == "weight_position":
            weighted_position_validity = position_validity.any(dim=1)[:, None]
            weighted_position = (p * weight[:, :, None]).sum(dim=1)  # (bs, 3, n)
            # Make the invalid position outside the range of -1 to 1 (all invalid positions become 2)
            weighted_position = weighted_position * weighted_position_validity + 2 * ~weighted_position_validity
            feature = self.sample_feature(tri_plane_feature[:, :32 * 3], weighted_position)  # (B, 32, n)
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
        # generate tri-plane feature conditioned on z and bone_length
        encoded_length = encode(bone_length, self.num_frequency_for_other, num_bone=self.num_bone_param)
        tri_plane_feature = self.tri_plane_gen(z, encoded_length[:, :, 0],
                                               truncation_psi=truncation_psi)  # (B, (32 + n_bone) * 3, h, w)
        return tri_plane_feature


class SSONARF(NARFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=False, parent=None, num_bone_param=None, view_dependent: bool = True):
        super(SSONARF, self).__init__()
        assert num_bone_param is not None
        assert hasattr(config, "origin_location")

        self.tri_plane_based = False
        self.config = config
        hidden_size = config.hidden_size
        # use_world_pose = not config.no_world_pose
        # use_ray_direction = not config.no_ray_direction
        # self.final_activation = config.final_activation
        self.origin_location = config.origin_location
        self.coordinate_scale = config.coordinate_scale
        # assert self.final_activation in ["tanh", "l2", None]
        assert self.origin_location in ["center", "center_fixed", "center+head"]
        assert parent is not None

        # dim = 3  # xyz
        # num_mlp_layers = 3
        # self.out_dim = config.out_dim if "out_dim" in self.config else 3
        self.parent_id = parent
        self.use_bone_length = bone_length
        # self.mask_before_PE = False
        # self.group_conv_first = config.group_conv_first

        # self.mask_input = self.config.concat and self.config.mask_input
        # self.selector_activation = self.config.selector_activation
        # selector_tmp = self.config.selector_adaptive_tmp.start
        # self.register_buffer("selector_tmp", torch.tensor(selector_tmp).float())

        self.density_activation = MyReLU.apply

        # self.density_scale = config.density_scale

        # parameters for position encoding
        nffp = self.config.num_frequency_for_position if "num_frequency_for_position" in self.config else 10
        nffo = self.config.num_frequency_for_other if "num_frequency_for_other" in self.config else 4

        self.num_frequency_for_position = nffp
        self.num_frequency_for_other = nffo

        self.hidden_size = hidden_size
        self.num_bone = num_bone - 1
        self.num_bone_param = num_bone_param if num_bone_param is not None else num_bone

        if type(z_dim) == list:
            self.z_dim = z_dim[0]
            self.z2_dim = z_dim[1]
        else:
            self.z_dim = z_dim
            self.z2_dim = z_dim

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
        self.view_dependent = view_dependent
        if view_dependent:
            self.mlp = StyledMLP(self.hidden_size + 3 * nffo * 2, self.hidden_size // 2,
                                 3, style_dim=self.z2_dim)
        else:
            self.mlp = StyledMLP(self.hidden_size, self.hidden_size // 2, 3, style_dim=self.z2_dim)

        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.temporal_state = {}

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
