from typing import Optional

import numpy as np
import torch
from torch import nn

from libraries.NARF.mesh_rendering import create_mesh
from libraries.NeRF.ray_sampler import mask_based_sampler, whole_image_grid_ray_sampler
from libraries.custom_stylegan2.net import Generator as StyleGANGenerator
from libraries.custom_stylegan2.net import PretrainedStyleGAN
from models.narf import TriPlaneNARF, MLPNARF


class TriNARFGenerator(nn.Module):  # tri-plane nerf
    def __init__(self, config, size, num_bone=1, parent_id=None, num_bone_param=None, black_background=False):
        super(TriNARFGenerator, self).__init__()
        self.config = config
        self.size = size
        self.num_bone = num_bone
        self.ray_sampler = whole_image_grid_ray_sampler
        self.background_ratio = config.background_ratio
        self.black_background = black_background

        z_dim = config.z_dim
        crop_background = config.crop_background

        if config.nerf_params.origin_location == "center+head":
            num_bone_param = num_bone
        self.nerf = TriPlaneNARF(config.nerf_params, z_dim=[z_dim * 2, z_dim], num_bone=num_bone,
                                 bone_length=True,
                                 parent=parent_id, num_bone_param=num_bone_param)
        if not black_background:
            if config.pretrained_background:
                self.background_generator = PretrainedStyleGAN()
            else:
                self.background_generator = StyleGANGenerator(size=size, style_dim=z_dim,
                                                              n_mlp=4, last_channel=3,
                                                              crop_background=crop_background)

    def register_canonical_pose(self, pose: np.ndarray):
        self.nerf.register_canonical_pose(pose)

    def normalized_inv_intrinsics(self, intrinsics: torch.tensor):
        normalized_intrinsics = torch.cat([intrinsics[:2] / self.size, intrinsics.new([[0, 0, 1]])], dim=0)
        normalized_inv_intri = torch.linalg.inv(normalized_intrinsics)
        return normalized_inv_intri

    @property
    def memory_cost(self):
        return self.nerf.memory_cost

    @property
    def flops(self):
        return self.nerf.flops

    def forward(self, pose_to_camera, pose_to_world, bone_length, z=None, inv_intrinsics=None,
                return_intermediate=False, truncation_psi=1, black_bg_if_possible=False, return_disparity=False,
                return_bg=False):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param pose_to_world: wold coordinate of joint
        :param bone_length:
        :param z: latent vector
        :param inv_intrinsics:
        :param return_intermediate:
        :return:
        """
        assert self.num_bone == 1 or (bone_length is not None and pose_to_camera is not None)
        batchsize = pose_to_camera.shape[0]

        grid, homo_img = self.ray_sampler(self.size, self.size, batchsize)

        if not self.black_background:
            z_dim = z.shape[1] // 4
            z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)
        else:
            z_dim = z.shape[1] // 3
            z_for_nerf, z_for_neural_render = torch.split(z, [z_dim * 2, z_dim], dim=1)

        # sparse rendering
        inv_intrinsics = torch.tensor(inv_intrinsics).float().cuda(homo_img.device)
        nerf_output = self.nerf(batchsize, homo_img,
                                pose_to_camera, inv_intrinsics, z_for_nerf,
                                z_for_neural_render,
                                bone_length,
                                Nc=self.config.nerf_params.Nc,
                                Nf=self.config.nerf_params.Nf,
                                return_intermediate=return_intermediate, truncation_psi=truncation_psi,
                                return_disparity=return_disparity)

        fg_color, fg_mask = nerf_output[:2]
        fine_weights = self.nerf.buffers_tensors["fine_weights"]
        fine_depth = self.nerf.buffers_tensors["fine_depth"]

        fg_color = fg_color.reshape(batchsize, 3, self.size, self.size)
        fg_mask = fg_mask.reshape(batchsize, self.size, self.size)

        if not self.black_background and not black_bg_if_possible:
            n_latent = self.background_generator.n_latent
            bg_color, _ = self.background_generator([z_for_background, z_for_neural_render], inject_index=n_latent - 4)
        else:
            bg_color = -1

        rendered_color = fg_color + (1 - fg_mask[:, None]) * bg_color

        if return_intermediate:
            fine_points, fine_density = nerf_output[-1]
            return rendered_color, fg_mask, fine_points, fine_density
        if return_disparity:
            disparity = nerf_output[2]
            disparity = disparity * self.config.nerf_params.coordinate_scale
            return rendered_color, fg_mask, disparity
        if return_bg:
            return fg_color, fg_mask, bg_color
        return rendered_color, fg_mask, fine_weights, fine_depth

    def render_mesh(self, pose_to_camera, intrinsics, z, bone_length, voxel_size=0.003,
                    mesh_th=15, truncation_psi=0.4):
        if not self.black_background:
            z_dim = z.shape[1] // 4
            z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)
        else:
            z_dim = z.shape[1] // 3
            z_for_nerf, z_for_neural_render = torch.split(z, [z_dim * 2, z_dim], dim=1)
        return self.nerf.render_mesh(pose_to_camera, intrinsics, z_for_nerf, z_for_neural_render, bone_length,
                                     voxel_size, mesh_th, truncation_psi, self.size)

    def create_mesh(self, pose_to_camera, z, bone_length, voxel_size=0.003,
                    mesh_th=15, truncation_psi=0.4):
        if not self.black_background:
            z_dim = z.shape[1] // 4
            z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)
        else:
            z_dim = z.shape[1] // 3
            z_for_nerf, z_for_neural_render = torch.split(z, [z_dim * 2, z_dim], dim=1)
        return create_mesh(self.nerf, pose_to_camera, z_for_nerf, z_for_neural_render, bone_length,
                           voxel_size, mesh_th, truncation_psi)


class DSONARFGenerator(nn.Module):
    def __init__(self, config, size, num_bone=1, parent_id=None, num_bone_param=None):
        super(DSONARFGenerator, self).__init__()
        self.config = config
        self.size = size
        self.num_bone = num_bone
        self.ray_sampler = mask_based_sampler

        nerf = TriPlaneNARF if config.use_triplane else MLPNARF

        self.time_conditional = self.config.nerf_params.time_conditional
        self.pose_conditional = self.config.nerf_params.pose_conditional

        z_dim = 0
        if self.time_conditional:
            z_dim += 20
        if self.pose_conditional:
            z_dim += (num_bone - 1) * 9

        if config.nerf_params.origin_location == "center+head":
            num_bone_param = num_bone
        view_dependent = not config.nerf_params.no_ray_direction
        self.nerf = nerf(config.nerf_params, z_dim=z_dim, num_bone=num_bone, bone_length=True,
                         parent=parent_id, num_bone_param=num_bone_param, view_dependent=view_dependent)

    def register_canonical_pose(self, pose: np.ndarray):
        if hasattr(self.nerf, "register_canonical_pose"):
            self.nerf.register_canonical_pose(pose)

    @property
    def memory_cost(self):
        return self.nerf.memory_cost

    @property
    def flops(self):
        return self.nerf.flops

    @staticmethod
    def positional_encoding(x: torch.Tensor, num_frequency: int) -> torch.Tensor:
        """
        positional encoding
        :param x: (B, )
        :param num_frequency: L in nerf paper
        :return:(B, n_freq * 2)
        """
        x = x[:, None] * 2 ** torch.arange(num_frequency, device=x.device) * np.pi
        encoded = torch.cat([torch.cos(x), torch.sin(x)], dim=1)
        return encoded

    @staticmethod
    def pose_encoding(pose: torch.Tensor):
        """

        :param pose: (B, num_bone, 4, 4)
        :return: (B, (num_bone - 1) * 3)
        """
        rot = pose[:, 1:, :3, :3]
        root_rot = pose[:, :1, :3, :3]

        encoded = torch.matmul(root_rot.permute(0, 1, 3, 2), rot)  # (B, num_bone - 1, 3, 3)
        return encoded.reshape(encoded.shape[0], -1)  # (B, (num_bone - 1) * 9)

    def get_latents(self, frame_time: torch.Tensor, pose_to_camera: torch.Tensor):
        zs = []
        if self.time_conditional:
            zs.append(self.positional_encoding(frame_time, num_frequency=10))
        if self.pose_conditional:
            zs.append(self.pose_encoding(pose_to_camera))

        assert len(zs) > 0

        z = torch.cat(zs, dim=1)

        z1 = z2 = z
        return z1, z2

    def forward(self, pose_to_camera, camera_pose, mask, frame_time, bone_length, inv_intrinsics,
                background: Optional[float] = None):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param camera_pose: camera rotation
        :param mask: foreground mask of object, (B, img_size, img_size)
        :param frame_time: normalized time of frame
        :param bone_length:
        :param background: background color
        :param inv_intrinsics:
        :return:
        """
        assert bone_length is not None and pose_to_camera is not None
        assert isinstance(inv_intrinsics, torch.Tensor)
        batchsize = pose_to_camera.shape[0]
        ray_batchsize = self.config.ray_batchsize

        grid, img_coord = self.ray_sampler(mask, ray_batchsize)

        # sparse rendering
        z1, z2 = self.get_latents(frame_time, pose_to_camera)

        # TODO: randomly replace z1 during training
        rendered_color, rendered_mask = self.nerf(batchsize, img_coord,
                                                  pose_to_camera, inv_intrinsics, z1, z2,
                                                  bone_length,
                                                  Nc=self.config.nerf_params.Nc,
                                                  Nf=self.config.nerf_params.Nf,
                                                  return_intermediate=False,
                                                  camera_pose=camera_pose,
                                                  )

        if background is None:
            background = -1
        rendered_color = rendered_color + background * (1 - rendered_mask[:, None])
        return rendered_color, rendered_mask, grid

    def render_entire_img(self, pose_to_camera, inv_intrinsics, frame_time, bone_length,
                          camera_pose=None, render_size=128,
                          semantic_map=False, use_normalized_intrinsics=False, no_grad=True, bbox=None):
        """

        :param pose_to_camera:
        :param inv_intrinsics:
        :param frame_time:
        :param bone_length:
        :param camera_pose:
        :param render_size:
        :param semantic_map:
        :param use_normalized_intrinsics:
        :param no_grad:
        :param bbox:
        :return:
        """
        # sparse rendering
        z1, z2 = self.get_latents(frame_time, pose_to_camera)
        return self.nerf.render_entire_img(pose_to_camera, inv_intrinsics, z1, z2, bone_length,
                                           camera_pose, render_size, self.config.nerf_params.Nc,
                                           self.config.nerf_params.Nf, semantic_map,
                                           use_normalized_intrinsics, no_grad=no_grad, bbox=bbox)

    def profile_memory_stats(self, pose_to_camera, inv_intrinsics, frame_time, bone_length,
                             camera_pose=None, render_size=128,
                             semantic_map=False, use_normalized_intrinsics=False):
        """

        :param pose_to_camera:
        :param inv_intrinsics:
        :param frame_time:
        :param bone_length:
        :param camera_pose:
        :param render_size:
        :param semantic_map:
        :param use_normalized_intrinsics:
        :return:
        """
        # sparse rendering
        z1, z2 = self.get_latents(frame_time, pose_to_camera)
        return self.nerf.profile_memory_stats(pose_to_camera, inv_intrinsics, z1, z2, bone_length,
                                              camera_pose, render_size, self.config.nerf_params.Nc,
                                              self.config.nerf_params.Nf, semantic_map,
                                              use_normalized_intrinsics)
