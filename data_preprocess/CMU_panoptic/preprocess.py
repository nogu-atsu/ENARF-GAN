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
from dependencies.smpl_utils import get_pose


def read_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame


def read_annots(video_path):
    ann_path = video_path[:-4] + "_info.mat"
    mat = scipy.io.loadmat(ann_path)
    return mat


def preprocess(path, thin_out_rate=10):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding='latin1')
    poses = data['poses'].reshape(-1, 24, 3)
    betas = data['betas'][None]

    poses = poses[::thin_out_rate]
    zrot = np.random.uniform(0, np.pi * 2, size=(len(poses)))
    n_pose = len(poses)

    # smpl forward
    A = get_pose(SMPL_MODEL, body_pose=torch.tensor(poses[:, 1:, :]).float(),
                 betas=torch.tensor(betas).float(),
                 global_orient=torch.tensor(poses[:, 0:1, :]).float()).numpy().copy()
    z = np.zeros_like(zrot)
    o = np.ones_like(zrot)
    trans = np.stack([np.cos(zrot), -np.sin(zrot), z, z,
                      np.sin(zrot), np.cos(zrot), z, z,
                      z, z, o, z,
                      z, z, z, o], axis=1).reshape(n_pose, 1, 4, 4)
    A_new = np.matmul(trans, A)  # (n_pose, 24, 4, 4)

    shift = A_new[:, [1, 2], :3, 3].transpose(1, 0, 2).mean(axis=1, keepdims=True)

    A_new[:, :, :3, 3] -= shift

    # axis_transform
    A_new = A_new[:, :, [1, 2, 0, 3]] * np.array([-1, -1, -1, 1])[:, None]

    # distance ~ N(7, 1)
    A_new[:, :, 2, 3] += np.clip(np.random.normal(7, 1, size=len(A_new)), a_min=3, a_max=None)[:, None]

    K = np.array([[600, 0, CROP_SIZE / 2],
                  [0, 600, CROP_SIZE / 2],
                  [0, 0, 1]], dtype="float")
    K[:2] *= IMG_SIZE / CROP_SIZE
    K = np.broadcast_to(K[None], (len(A_new), 3, 3))

    return K, A_new


if __name__ == "__main__":
    IMG_SIZE = 128
    CROP_SIZE = 180
    SMPL_MODEL = SMPL(model_path="../../smpl_data", gender="neutral")
    DATA_ROOT = "/data/unagi0/noguchi/dataset/mosh/neutrMosh/"

    smpl_path = glob.glob(f"{DATA_ROOT}/neutrSMPL_CMU/*/*.pkl")
    print(len(smpl_path))

    pose_cache = []
    intrinsic_cache = []
    for path in tqdm(smpl_path):
        K, pose_3d = preprocess(path)
        pose_cache.append(pose_3d)
        intrinsic_cache.append(K)
    pose_cache = np.concatenate(pose_cache)
    intrinsic_cache = np.concatenate(intrinsic_cache)
    print("pose shape", pose_cache.shape)
    cache = {"camera_intrinsic": intrinsic_cache,
             "smpl_pose": pose_cache}

    os.makedirs(f"{DATA_ROOT}/NARF_GAN_cache", exist_ok=True)
    with open(f"{DATA_ROOT}/NARF_GAN_cache/cache.pickle", "wb") as f:
        pickle.dump(cache, f)

    np.save(f"{DATA_ROOT}/NARF_GAN_cache/canonical.npy",
            np.load("../../smpl_data/neutral_canonical.npy"))
