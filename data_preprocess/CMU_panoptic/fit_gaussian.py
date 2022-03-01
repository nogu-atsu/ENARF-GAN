import glob
import os
import pickle
import sys

import cv2
import numpy as np
import scipy.io
import torch
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../../")
from utils.smpl_utils import get_pose


def read_smpl(path, thin_out_rate=10):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding='latin1')
    poses = data['poses'].reshape(-1, 24, 3)
    betas = data['betas'][None]

    return poses, betas


def main():
    smpl_path = glob.glob(f"{DATA_ROOT}/neutrSMPL_CMU/*/*.pkl")
    print(len(smpl_path))

    all_poses = []
    all_betas = []
    for path in tqdm(smpl_path):
        poses, betas = read_smpl(path)
        all_poses.append(poses)
        all_betas.append(betas)

    all_poses = np.concatenate(all_poses)  # (n, 72)
    all_betas = np.concatenate(all_betas)  # (n,)

    mean_pose = np.mean(all_poses, axis=0)
    mean_beta = np.mean(all_betas, axis=0)

    covs_pose = []
    for i in range(24):
        covs_pose.append(np.cov(all_poses[:, i].T))
    covs_beta = np.cov(all_betas.T)

    n_sample = 400000
    sampled_betas = np.random.multivariate_normal(mean_beta, covs_beta, n_sample)

    sampled_poses = []
    for i in range(24):
        p = np.random.multivariate_normal(mean_pose[i], covs_pose[i], n_sample)
        sampled_poses.append(p)
    sampled_poses = np.stack(sampled_poses, axis=1)

    sampled_pose_3d = get_pose(SMPL_MODEL, body_pose=torch.tensor(sampled_poses[:, 1:, :]).float(),
                               betas=torch.tensor(sampled_betas).float(),
                               global_orient=torch.tensor(sampled_poses[:, 0:1, :] * 0).float()).numpy()

    right_center = sampled_pose_3d[:, [7, 10], :3, 3].transpose(1, 0, 2).mean(axis=1)
    left_center = sampled_pose_3d[:, [8, 11], :3, 3].transpose(1, 0, 2).mean(axis=1)
    eps = np.random.uniform(size=len(right_center))[:, None]
    foot_middle = right_center * eps + left_center * (1 - eps)
    z_axis = sampled_pose_3d[:, 0, :3, 3] - foot_middle
    z_axis = z_axis / np.linalg.norm(z_axis, axis=1)[:, None]
    new_axis = np.array([1, 0, 0])
    y_axis = new_axis - np.dot(z_axis, np.array([1, 0, 0]))[:, None] * z_axis
    y_axis = y_axis / np.linalg.norm(z_axis, axis=1)[:, None]
    x_axis = np.cross(y_axis, z_axis)

    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    eye = np.eye(4)[None].repeat(n_sample, axis=0).copy()
    eye[:, :3, :3] = rot

    sampled_pose_3d = np.matmul(eye[:, None], sampled_pose_3d)

    n_pose = len(sampled_pose_3d)
    zrot = np.random.uniform(0, np.pi * 2, size=(n_pose, ))
    z = np.zeros_like(zrot)
    o = np.ones_like(zrot)
    trans = np.stack([np.cos(zrot), -np.sin(zrot), z, z,
                      np.sin(zrot), np.cos(zrot), z, z,
                      z, z, o, z,
                      z, z, z, o], axis=1).reshape(n_pose, 1, 4, 4)

    sampled_pose_3d = np.matmul(trans, sampled_pose_3d)  # (n_pose, 24, 4, 4)

    shift = sampled_pose_3d[:, [1, 2], :3, 3].transpose(1, 0, 2).mean(axis=1, keepdims=True)

    sampled_pose_3d[:, :, :3, 3] -= shift

    # axis_transform
    sampled_pose_3d = sampled_pose_3d[:, :, [1, 2, 0, 3]] * np.array([-1, -1, -1, 1])[:, None]

    # distance ~ N(7, 1)
    sampled_pose_3d[:, :, 2, 3] += np.clip(np.random.normal(7, 1, size=len(sampled_pose_3d)),
                                           a_min=3, a_max=None)[:, None]

    K = np.array([[600, 0, CROP_SIZE / 2],
                  [0, 600, CROP_SIZE / 2],
                  [0, 0, 1]], dtype="float")
    K[:2] *= IMG_SIZE / CROP_SIZE
    K = np.broadcast_to(K[None], (len(sampled_pose_3d), 3, 3))

    cache = {"camera_intrinsic": K,
             "smpl_pose": sampled_pose_3d}

    os.makedirs(f"{DATA_ROOT}/NARF_GAN_gaussian_cache", exist_ok=True)
    with open(f"{DATA_ROOT}/NARF_GAN_gaussian_cache/cache.pickle", "wb") as f:
        pickle.dump(cache, f)

    np.save(f"{DATA_ROOT}/NARF_GAN_gaussian_cache/canonical.npy",
            np.load("../../smpl_data/neutral_canonical.npy"))


if __name__ == "__main__":
    IMG_SIZE = 128
    CROP_SIZE = 180
    SMPL_MODEL = SMPL(model_path="../../smpl_data", gender="neutral")
    DATA_ROOT = "/data/unagi0/noguchi/dataset/mosh/neutrMosh/"

    main()
