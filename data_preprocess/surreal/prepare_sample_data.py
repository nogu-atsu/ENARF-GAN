import argparse
import glob
import os
import pickle
import sys

import numpy as np
import scipy.io
import torch
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../")
from dependencies.smpl_utils import get_pose
from data_preprocess.utils import get_bone_length, SMPL_PARENTS

IMG_SIZE = 128
CROP_SIZE = 180
SMPL_MODEL = {"male": SMPL(model_path="../smpl_data", gender="male"),
              "female": SMPL(model_path="../smpl_data", gender="female")}


def read_annots(video_path):
    ann_path = video_path[:-4] + "_info.mat"
    mat = scipy.io.loadmat(ann_path)
    return mat


def read_pose_and_crop(path):
    annot = read_annots(path)
    gender = ["female", "male"][annot["gender"][0, 0]]
    poses = annot["pose"]
    poses = poses[:, 0].reshape(1, 24, 3)
    betas = annot["shape"][None, :, 0]
    zrot = annot["zrot"][0, 0]

    # smpl forward
    A = get_pose(SMPL_MODEL[gender], body_pose=torch.tensor(poses[:, 1:, :]).float(),
                 betas=torch.tensor(betas).float(),
                 global_orient=torch.tensor(poses[:, 0:1, :]).float()).numpy().copy()

    trans = np.array([[np.cos(zrot), -np.sin(zrot), 0, 0],
                      [np.sin(zrot), np.cos(zrot), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    A_new = np.matmul(trans, A)

    if annot["joints3D"].ndim != 3:
        return (None,) * 6
    # shift
    joints3D = annot["joints3D"][:, :, 0]
    camLoc = annot["camLoc"]
    j3D = (joints3D - camLoc).mean(axis=1)
    j3D = j3D * np.array([1, -1, 1])
    j3D = j3D[[0, 2, 1]]

    A_mean = A_new[0, :, :3, 3].mean(axis=0)

    shift = j3D - A_mean
    A_new[:, :, :3, 3] += shift

    # axis_transform
    A_new = A_new[:, :, [1, 2, 0, 3]] * np.array([-1, -1, -1, 1])[:, None]

    K = np.array([[600, 0, 160],
                  [0, 600, 120],
                  [0, 0, 1]], dtype="float")

    pose_3d = A_new[0, :, :3, 3:]
    pose_2d = np.matmul(K, pose_3d)
    pose_2d = pose_2d[:, :2, 0] / pose_2d[:, 2:, 0]
    # crop
    center = pose_2d[[1, 2]].mean(axis=0).astype("int")
    x1, x2 = center[0] - CROP_SIZE // 2, center[0] + CROP_SIZE // 2
    y1, y2 = center[1] - CROP_SIZE // 2, center[1] + CROP_SIZE // 2

    cropped_K = K.copy()
    cropped_K[:2, 2] -= np.array([x1, y1])
    resized_K = cropped_K.copy()
    resized_K[:2] *= IMG_SIZE / CROP_SIZE
    return x1, x2, y1, y2, A_new, resized_K


def preprocess(path):
    x1, x2, y1, y2, A_new, resized_K = read_pose_and_crop(path)
    if x1 is None:
        return None, None

    return resized_K, A_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    DATA_ROOT = args.data_path

    video_path = glob.glob(f"{DATA_ROOT}/val/run0/*/*.mp4")
    print(len(video_path))

    sample_data = []
    for path in tqdm(video_path[:5]):
        K, pose_3d = preprocess(path)

        sample_data.append({
            "pose_to_camera": pose_3d[0],
            "intrinsics": K,
            "bone_length": get_bone_length(pose_3d[0], SMPL_PARENTS)
        })

    out_dir = f"../data/surreal"
    os.makedirs(out_dir)
    with open(f"{out_dir}/sample_data.pickle", "wb") as f:
        pickle.dump(sample_data, f)
