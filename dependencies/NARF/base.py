from typing import Optional, Union, List

import torch

from dependencies.NARF.mesh_rendering import render_mesh_, create_mesh
from dependencies.NARF.pose_utils import transform_pose
from dependencies.NeRF.base import NeRFBase
from dependencies.NeRF.rendering import render_entire_img


class NARFBase(NeRFBase):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=True, parent=None, num_bone_param=None,
                 view_dependent: bool = False):
        super(NARFBase, self).__init__(config, z_dim, view_dependent)
        self.num_bone = num_bone - 1 if self.origin_location in ["center", "center_fixed"] else num_bone
        self.use_bone_length = bone_length
        assert parent is not None
        self.parent_id = parent

    def transform_pose(self, pose_to_camera: torch.Tensor, bone_length: torch.Tensor):
        pose_to_camera, bone_length = transform_pose(pose_to_camera, bone_length,
                                                     self.origin_location, self.parent_id)
        return pose_to_camera, bone_length

    def forward(self, batchsize, sampled_img_coord, pose_to_camera, inv_intrinsics,
                z, z_rend, bone_length,
                render_scale=1, Nc=64, Nf=128, return_intermediate=False,
                truncation_psi=1,
                camera_pose: Optional[torch.Tensor] = None,
                return_disparity=False):
        """
        rendering function for sampled rays
        :param sampled_img_coord: sampled image coordinate
        :param pose_to_camera:
        :param inv_intrinsics:
        :param render_scale:
        :param Nc:
        :param Nf:
        :param return_intermediate:
        :param truncation_psi:
        :param camera_pose:
        :param return_disparity:
        :return: color and mask value for sampled rays
        """
        # TODO: triplane narfと関数をまとめる
        model_input = {"z": z, "z_rend": z_rend, "bone_length": bone_length, "truncation_psi": truncation_psi}
        pose_to_camera, model_input["bone_length"] = self.transform_pose(pose_to_camera,
                                                                         model_input["bone_length"])
        return self._forward(sampled_img_coord, pose_to_camera, inv_intrinsics,
                             render_scale, Nc, Nf, return_intermediate,
                             camera_pose, model_input, return_disparity)

    def render_entire_img(self, pose_to_camera, inv_intrinsics, z, z_rend, bone_length, camera_pose=None,
                          render_size=128, Nc=64, Nf=128, semantic_map=False, use_normalized_intrinsics=False,
                          no_grad=True, truncation_psi=1, ):
        model_input = {"z": z, "z_rend": z_rend, "bone_length": bone_length, "truncation_psi": truncation_psi}
        pose_to_camera, model_input["bone_length"] = self.transform_pose(pose_to_camera,
                                                                         model_input["bone_length"])
        if self.tri_plane_based:
            model_input["tri_plane_feature"] = self.compute_tri_plane_feature(z, bone_length)
        return render_entire_img(self, pose_to_camera, inv_intrinsics, camera_pose,
                                 render_size, Nc, Nf, semantic_map, use_normalized_intrinsics,
                                 no_grad, model_input)

    def render_mesh(self, pose_to_camera, intrinsics, z, z_rend, bone_length, voxel_size=0.003,
                    mesh_th=15, truncation_psi=0.4, img_size=128):
        raise NotImplementedError("new mesh rendering is not implemented")
        assert z is None or z.shape[0] == 1
        assert bone_length is None or bone_length.shape[0] == 1

        meshes = create_mesh(self, pose_to_camera, z, z_rend, bone_length,
                             voxel_size=voxel_size,
                             mesh_th=mesh_th, truncation_psi=truncation_psi)

        images = render_mesh_(meshes, intrinsics, img_size)

        return images, meshes
