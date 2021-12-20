from typing import Optional

import cv2
import numpy as np
import torch
from smplx.lbs import (blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform)
from smplx.utils import Tensor


def get_pose(
        smpl,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        pose2rot: bool = True,
        **kwargs
) -> Tensor:
    ''' Forward pass for the SMPL model
        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3
            If given, ignore the member variable and use it as the global
            rotation of the body. Useful if someone wishes to predicts this
            with an external model. (default=None)
        betas: torch.tensor, optional, shape BxN_b
            If given, ignore the member variable `betas` and use it
            instead. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        Returns
        -------
    '''
    # If no shape and pose parameters are passed along, then use the
    # ones from the module
    global_orient = (global_orient if global_orient is not None else
                     smpl.global_orient)
    body_pose = body_pose if body_pose is not None else smpl.body_pose
    betas = betas if betas is not None else smpl.betas

    full_pose = torch.cat([global_orient, body_pose], dim=1)

    batch_size = max(betas.shape[0], global_orient.shape[0],
                     body_pose.shape[0])

    if betas.shape[0] != batch_size:
        num_repeats = int(batch_size / betas.shape[0])
        betas = betas.expand(num_repeats, -1)

    A = _get_pose(betas, full_pose, smpl.v_template,
                  smpl.shapedirs,
                  smpl.J_regressor, smpl.parents,
                  pose2rot=pose2rot)
    return A


def _get_pose(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        J_regressor: Tensor,
        parents: Tensor,
        pose2rot: bool = True,
) -> Tensor:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters
        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        # (N x P) x (P, V * 3) -> N x V x 3
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    A[:, :, :3, 3] = J_transformed
    return A


def move_to_origin(bone_pose, scale=0.5):
    """translates the model to the origin"""
    left_hip = 1
    right_hip = 2
    trans = -bone_pose[:, [left_hip, right_hip], :3, 3].mean(axis=1)
    bone_pose = (bone_pose + trans) * scale
    return bone_pose


def axis_transformation(bone_pose: np.ndarray, axis_transformation: np.ndarray = np.array([1, -1, -1])):
    bone_pose[:, :3] *= axis_transformation[None, :, None]
    return bone_pose


# def rotate_pose_in_place(pose, x_r, y_r, z_r):
#     """rotates model (x-axis first, then y-axis, and then z-axis)"""
#     mat_x, _ = cv2.Rodrigues(np.asarray([x_r, 0, 0], dtype=np.float32))
#     mat_y, _ = cv2.Rodrigues(np.asarray([0, y_r, 0], dtype=np.float32))
#     mat_z, _ = cv2.Rodrigues(np.asarray([0, 0, z_r], dtype=np.float32))
#     mat = np.matmul(np.matmul(mat_x, mat_y), mat_z)
#     T = np.eye(4)
#     T[:3, :3] = mat
#
#     pose = np.matmul(T, pose)
#
#     return pose
#
#
# def transform_model_randomly(bone_pose):
#     """translates the model to the origin, and rotates it randomly"""
#     # random rotation
#     y_rot = np.random.uniform(-np.pi, np.pi)
#     x_rot = np.random.uniform(-0.3, 0.3)
#     z_rot = np.random.uniform(-0.3, 0.3)
#     # random scale
#     scale = np.random.uniform(1.0, 1.5)
#     # # mul(mat, mesh.v)
#     bone_pose = rotate_pose_in_place(bone_pose, x_rot, y_rot, z_rot)
#
#     bone_pose = bone_pose * scale
#
#     # create a dict of transformation parameters
#     param = dict()
#     param['scale'] = scale
#     param['rot'] = np.array([x_rot, y_rot, z_rot])
#     return bone_pose, param
#
#
# def sample_camera_mat(cam_t=None, theta=None, phi=None, angle=None, deterministic=False):
#     if deterministic:
#         if cam_t is None:
#             cam_t = np.array((0, 0, 2.0))
#
#             theta = 0
#             phi = 0
#             angle = 0
#         cam_r = np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
#         cam_r = cam_r * angle
#     else:
#         if cam_t is None:
#             cam_t = np.array((0, 0, 2.0))
#
#             theta = np.random.uniform(0, 0.3)
#             phi = np.random.uniform(0, 2 * np.pi)
#             angle = np.random.uniform(0, 2 * np.pi)
#         cam_r = np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
#         cam_r = cam_r * angle
#
#     R = cv2.Rodrigues(cam_r)[0]
#     T = np.vstack((np.hstack((R, cam_t.reshape(3, 1))), np.zeros((1, 4))))  # 4 x 4
#
#     return T
