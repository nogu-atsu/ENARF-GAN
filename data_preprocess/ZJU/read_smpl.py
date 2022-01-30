# read smpl for zju

import cv2
import numpy as np
import torch
from easymocap.smplmodel import load_model
from easymocap.smplmodel.lbs import (batch_rigid_transform, batch_rodrigues,
                                     blend_shapes, vertices2joints, lbs)


def extract_bone(betas, pose, v_template, shapedirs, J_regressor, parents,
                 pose2rot=True, dtype=torch.float32):
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
        dtype: torch.dtype, optional
        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    return J_transformed, A


class PoseLoader:
    def __init__(self, smpl_model_path):
        self.body_model = load_model(
            gender="neutral",
            model_type="smplx",
            model_path=smpl_model_path,
            device="cpu")
        self.joints_to_use = np.array(list(range(22)) + [28, 43])

    def __call__(self, smpl_param):
        Rh = np.array(smpl_param["Rh"])  # 1 x 3
        Th = np.array(smpl_param["Th"])  # 1 x 3
        poses = np.array(smpl_param["poses"])  # 1 x 87
        shapes = smpl_param["shapes"]  # 1 x 10
        expression = smpl_param["expression"]  # 1 x 10

        shapes = torch.tensor(shapes).float()
        expression = torch.tensor(expression).float()
        shapes = torch.cat([shapes, expression], dim=1)
        poses = torch.tensor(poses).float()
        poses = self.body_model.extend_pose(poses)
        v_template = self.body_model.j_v_template
        joints, A = extract_bone(shapes, poses, v_template,
                                 self.body_model.j_shapedirs,
                                 self.body_model.j_J_regressor, self.body_model.parents,
                                 pose2rot=True, dtype=self.body_model.dtype)

        bone_pose = A.clone()
        bone_pose[:, :, :3, 3] = joints

        trans = np.eye(4)
        trans[:3, :3] = cv2.Rodrigues(Rh[0])[0]
        trans[:3, 3] = Th

        bone_pose_world = np.matmul(trans, bone_pose.numpy()[0])
        return bone_pose_world[self.joints_to_use]

    def canonical_pose(self):
        # Rh = np.zeros((1, 3))  # 1 x 3
        # Th = np.zeros((1, 3))  # 1 x 3
        poses = np.zeros((1, 87))  # 1 x 87
        # shapes = np.zeros((1, 10))  # 1 x 10
        expression = np.zeros((1, 10))  # 1 x 10

        shapes = torch.zeros(1, 10).float()
        expression = torch.zeros(1, 10).float()
        shapes = torch.cat([shapes, expression], dim=1)
        poses = torch.zeros(1, 87).float()
        poses = self.body_model.extend_pose(poses)
        v_template = self.body_model.j_v_template
        joints, A = extract_bone(shapes, poses, v_template,
                                 self.body_model.j_shapedirs,
                                 self.body_model.j_J_regressor, self.body_model.parents,
                                 pose2rot=True, dtype=self.body_model.dtype)

        bone_pose = A.clone()
        bone_pose[:, :, :3, 3] = joints

        # trans = np.eye(4)
        # trans[:3, :3] = cv2.Rodrigues(Rh[0])[0]
        # trans[:3, 3] = Th

        bone_pose_world = bone_pose.numpy()[0]
        # move origin
        bone_pose_world = bone_pose_world - bone_pose_world[[1, 2]].mean(axis=0)
        return bone_pose_world[self.joints_to_use]
