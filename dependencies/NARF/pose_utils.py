from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def rotation_matrix(theta: torch.tensor):
    batchsize = theta.shape[0]
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    R = torch.stack([c, z, -s, z,
                     z, o, z, z,
                     s, z, c, z,
                     z, z, z, o], dim=-1)  # (B, 16)
    R = R.reshape(batchsize, 4, 4)
    return R


def rotate_pose_randomly(pose_3d: torch.tensor):
    batchsize = pose_3d.shape[0]
    rotate_angle = pose_3d.new_empty((batchsize,)).uniform_(0, 2 * np.pi)
    R = rotation_matrix(rotate_angle)

    rotated_pose_3d = rotate_pose(pose_3d, R)
    return rotated_pose_3d


def rotate_pose_by_angle(pose_3d: torch.tensor, angle):
    R = rotation_matrix(angle)
    rotated_pose_3d = rotate_pose(pose_3d, R)
    return rotated_pose_3d


def rotate_pose(pose_3d: torch.tensor, R: torch.tensor):
    zeros_33 = torch.zeros(pose_3d.shape[0], 3, 3, device=R.device, dtype=torch.float)
    center = torch.cat([zeros_33, pose_3d[:, :, :3, 3:].mean(dim=1)], dim=-1)
    zeros_14 = torch.zeros(pose_3d.shape[0], 1, 4, device=R.device, dtype=torch.float)
    center = torch.cat([center, zeros_14], dim=1)[:, None]
    pose_camera_theta = torch.matmul(R[:, None], (pose_3d - center)) + center
    return pose_camera_theta


def interpolate_pose(pose_3d: np.ndarray, parents: np.ndarray, num: int = 100, loop: bool = True):
    """linear interpolation among poses in pose_3d

    Args:
        pose_3d: (num_pose, n_parts, 4, 4)
        parents: (n_parts, )
        num: number of output poses
        loop:

    Returns:

    """
    num_pose, num_parts, _, _ = pose_3d.shape

    parent_mat = pose_3d[:, parents[1:]]  # num_pose x 23 x 4 x 4
    parent_mat = np.concatenate([np.tile(np.eye(4)[None, None], (num_pose, 1, 1, 1)), parent_mat], axis=1)

    child_translation = []
    for i in range(num_pose):
        trans_i = []
        for j in range(num_parts):
            trans_i.append(np.linalg.inv(parent_mat[i, j]).dot(pose_3d[i, j]))
        child_translation.append(np.array(trans_i))
    child_translation = np.array(child_translation)  # num_pose x 24 x 4 x 4

    # interpolation (slerp)
    interp_pose_to_world = []
    for i in range(num_parts):
        if loop:
            key_rots = np.concatenate([child_translation[:, i, :3, :3],
                                       child_translation[:1, i, :3, :3]], axis=0)  # repeat first
            key_times = np.arange(num_pose + 1)
            times = np.arange(num) * num_pose / num
            interp_trans = np.concatenate([
                np.linspace(child_translation[j, i, :3, 3],
                            child_translation[(j + 1) % num_pose, i, :3, 3],
                            num // num_pose, endpoint=False) for j in range(num_pose)], axis=0)  # num x 3
        else:
            key_rots = child_translation[:, i, :3, :3]
            key_times = np.arange(num_pose)
            times = np.arange(num) * (num_pose - 1) / (num - 1)
            interp_trans = np.concatenate([
                np.linspace(child_translation[j, i, :3, 3],
                            child_translation[(j + 1), i, :3, 3],
                            num // (num_pose - 1), endpoint=True) for j in range(num_pose - 1)], axis=0)  # num x 3
        slerp = Slerp(key_times, R.from_matrix(key_rots))
        interp_rots = slerp(times).as_matrix()  # num x 3 x 3

        interp_mat = np.concatenate([interp_rots, interp_trans[:, :, None]], axis=2)
        interp_mat = np.concatenate([interp_mat, np.tile(np.array([[[0, 0, 0, 1]]]), (num, 1, 1))],
                                    axis=1)  # num x 4 x 4
        interp_pose_to_world.append(interp_mat)
    interp_pose_to_world = np.array(interp_pose_to_world)  # num_parts x num x 4 x 4

    interpolated_poses = []
    for i in range(num):
        interp_pose = []
        for part_idx in range(num_parts):
            if parents[part_idx] == -1:
                mat = np.eye(4)
            else:
                mat = interp_pose[parents[part_idx]]
            mat = mat.dot(interp_pose_to_world[part_idx, i])

            interp_pose.append(mat)

        interpolated_poses.append(np.stack(interp_pose))
    return np.stack(interpolated_poses)


def rotate_mesh_by_angle(pose_3d: torch.Tensor, meshes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], angle):
    meshes = deepcopy(meshes)
    vertices = meshes[0]  # (V, 3)
    center = pose_3d[0, :, :3, 3:].mean(dim=0)  # (3, 1)
    R = rotation_matrix(angle)
    rotated_vertices = torch.matmul(R[0, :3, :3], (vertices.permute(1, 0) - center)) + R[0, :3, 3:] + center
    rotated_vertices = rotated_vertices.permute(1, 0)
    meshes = (rotated_vertices,) + meshes[1:]
    return meshes


def transform_pose(pose_to_camera, bone_length, origin_location, parent_id):
    if origin_location == "center":
        pose_to_camera = torch.cat([pose_to_camera[:, 1:, :, :3],
                                    (pose_to_camera[:, 1:, :, 3:] +
                                     pose_to_camera[:, parent_id[1:], :, 3:]) / 2], dim=-1)
    elif origin_location == "center_fixed":
        pose_to_camera = torch.cat([pose_to_camera[:, parent_id[1:], :, :3],
                                    (pose_to_camera[:, 1:, :, 3:] +
                                     pose_to_camera[:, parent_id[1:], :, 3:]) / 2], dim=-1)

    elif origin_location == "center+head":
        bone_length = torch.cat([bone_length, torch.ones(bone_length.shape[0], 1, 1, device=bone_length.device)],
                                dim=1)  # (B, 24)
        head_id = 15
        _pose_to_camera = torch.cat([pose_to_camera[:, parent_id[1:], :, :3],
                                     (pose_to_camera[:, 1:, :, 3:] +
                                      pose_to_camera[:, parent_id[1:], :, 3:]) / 2],
                                    dim=-1)  # (B, 23, 4, 4)
        pose_to_camera = torch.cat([_pose_to_camera, pose_to_camera[:, head_id][:, None]], dim=1)  # (B, 24, 4, 4)
    return pose_to_camera, bone_length
