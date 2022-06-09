import time
from typing import Optional

import torch
from torch import nn

from dependencies.NARF.mesh_rendering import render_mesh_, create_mesh
from dependencies.NARF.rendering import render, render_entire_img


class NARFBase(nn.Module):
    def __init__(self):
        super(NARFBase, self).__init__()

    def time_start(self):
        torch.cuda.synchronize()
        start = time.time()
        return start

    def time_end(self, name, start):
        torch.cuda.synchronize()
        end = time.time()
        if name not in self.buffers_tensors:
            self.buffers_tensors[name] = 0
        self.buffers_tensors[name] += end - start

    @property
    def memory_cost(self):
        raise NotImplementedError()

    @property
    def flops(self):
        raise NotImplementedError()

    def calc_color_and_density(self, local_pos: torch.Tensor, canonical_pos: torch.Tensor,
                               tri_plane_feature: torch.Tensor,
                               z_rend: torch.Tensor, bone_length: torch.Tensor, mode: str,
                               ray_direction: Optional[torch.Tensor] = None):
        """
        forward func of ImplicitField
        :param local_pos: local coordinate, (B * n_bone, 3, n) (n = num_of_ray * points_on_ray)
        :param canonical_pos: canonical coordinate, (B, n_bone, 3, n) (n = num_of_ray * points_on_ray)
        :param tri_plane_feature:
        :param z_rend: b x groups x 4 x 4
        :param bone_length: b x groups x 1
        :param mode: str
        :param ray_direction
        :return: b x groups x 4 x n
        """
        raise NotImplementedError()

    def calc_density_and_color_from_camera_coord(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                 bone_length: torch.Tensor, z, z_rend, ray_direction):
        """compute density from positions in camera coordinate

        :param position:
        :param pose_to_camera:
        :param bone_length:
        :param z:
        :param z_rend:
        :param ray_direction:
        :return: density of input positions
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def forward(self, batchsize, sampled_img_coord, pose_to_camera, inv_intrinsics, z, z_rend,
                bone_length, render_scale=1, Nc=64, Nf=128,
                return_intermediate=False, camera_pose: Optional[torch.Tensor] = None,
                truncation_psi=1, return_disparity=False):
        """
        rendering function for sampled rays
        :param batchsize:
        :param sampled_img_coord: sampled image coordinate
        :param pose_to_camera:
        :param inv_intrinsics:
        :param z:
        :param z_rend:
        :param bone_length:
        :param render_scale:
        :param Nc:
        :param Nf:
        :param return_intermediate:
        :param camera_pose:
        :return: color and mask value for sampled rays
        """
        assert not self.view_dependent or camera_pose is not None
        nerf_output = render(self, sampled_img_coord,
                             pose_to_camera,
                             inv_intrinsics,
                             z=z,
                             z_rend=z_rend,
                             bone_length=bone_length,
                             Nc=Nc,
                             Nf=Nf,
                             render_scale=render_scale,
                             return_intermediate=return_intermediate,
                             camera_pose=camera_pose,
                             truncation_psi=truncation_psi)
        if return_intermediate:
            merged_color, merged_mask, _, intermediate_output = nerf_output
            return merged_color, merged_mask, intermediate_output

        merged_color, merged_mask, merged_disparity = nerf_output
        if return_disparity:
            return merged_color, merged_mask, merged_disparity
        return merged_color, merged_mask

    def render_entire_img(self, pose_to_camera, inv_intrinsics, z, z_rend, bone_length, camera_pose=None,
                          render_size=128, Nc=64, Nf=128,
                          semantic_map=False, use_normalized_intrinsics=False, no_grad=True):
        return render_entire_img(self, pose_to_camera, inv_intrinsics, z, z_rend, bone_length,
                                 camera_pose=camera_pose, render_size=render_size, Nc=Nc, Nf=Nf,
                                 semantic_map=semantic_map,
                                 use_normalized_intrinsics=use_normalized_intrinsics, no_grad=no_grad)

    def render_mesh(self, pose_to_camera, intrinsics, z, z_rend, bone_length, voxel_size=0.003,
                    mesh_th=15, truncation_psi=0.4, img_size=128):
        assert z is None or z.shape[0] == 1
        assert bone_length is None or bone_length.shape[0] == 1

        meshes = create_mesh(self, pose_to_camera, z, z_rend, bone_length,
                             voxel_size=voxel_size,
                             mesh_th=mesh_th, truncation_psi=truncation_psi)

        images = render_mesh_(meshes, intrinsics, img_size)

        return images, meshes

    def profile_memory_stats(self, pose_to_camera, inv_intrinsics, z, z_rend, bone_length, camera_pose=None,
                             render_size=128, Nc=64, Nf=128,
                             semantic_map=False, use_normalized_intrinsics=False):
        """

        :param pose_to_camera:
        :param inv_intrinsics:
        :param z:
        :param z_rend:
        :param bone_length:
        :param camera_pose:
        :param render_size:
        :param Nc:
        :param Nf:
        :param semantic_map:
        :param use_normalized_intrinsics:
        :return:
        """
        assert z is None or z.shape[0] == 1
        assert bone_length is None or bone_length.shape[0] == 1
        ray_batchsize = self.config.render_bs

        from dependencies.memory_reporter import get_gpu_properties

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        properties = get_gpu_properties()
        print("model", list(properties)[5]['memory.used'])
        initial_memory = int(list(properties)[5]['memory.used'])
        if use_normalized_intrinsics:
            img_coord = torch.stack([(torch.arange(render_size * render_size) % render_size + 0.5) / render_size,
                                     (torch.arange(render_size * render_size) // render_size + 0.5) / render_size,
                                     torch.ones(render_size * render_size).long()], dim=0).float()
        else:
            img_coord = torch.stack([torch.arange(render_size * render_size) % render_size + 0.5,
                                     torch.arange(render_size * render_size) // render_size + 0.5,
                                     torch.ones(render_size * render_size).long()], dim=0).float()

        img_coord = img_coord[None, None].cuda()

        # # count rays
        # self.valid_rays = 0
        # self.valid_canonical_pos = 0

        if self.tri_plane_based:
            if self.origin_location == "center+head":
                _bone_length = torch.cat([bone_length,
                                          torch.ones(bone_length.shape[0], 1, 1, device=bone_length.device)],
                                         dim=1)  # (B, 24)
            else:
                _bone_length = bone_length
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            properties = get_gpu_properties()
            tri_plane_feature = self.compute_tri_plane_feature(z, _bone_length)
            initial_memory = int(list(properties)[5]['memory.used'])
        else:
            tri_plane_feature = None

        all_required_memory = initial_memory
        print("tri-plane", list(properties)[5]['memory.used'])
        for i in range(0, render_size ** 2, ray_batchsize):
            (rendered_color_i, rendered_mask_i,
             rendered_disparity_i) = render(self, img_coord[:, :, :, i:i + ray_batchsize],
                                            pose_to_camera[:1],
                                            inv_intrinsics,
                                            z=z,
                                            z_rend=z_rend,
                                            bone_length=bone_length,
                                            Nc=Nc,
                                            Nf=Nf,
                                            camera_pose=camera_pose,
                                            tri_plane_feature=tri_plane_feature)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            properties = get_gpu_properties()
            print("inference", list(properties)[5]['memory.used'])
            all_required_memory += int(list(properties)[5]['memory.used']) - initial_memory

        print("all required memory", all_required_memory, "MB")
