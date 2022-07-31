import argparse
import glob
import os
import pickle
import sys

import blosc
import cv2
import numpy as np
import scipy.io
import torch
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../")
from dependencies.smpl_utils import get_pose

IMG_SIZE = 128
CROP_SIZE = 180
SMPL_MODEL = {"male": SMPL(model_path="../smpl_data", gender="male"),
              "female": SMPL(model_path="../smpl_data", gender="female")}


def read_frame(video_path, return_mask=False):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if return_mask:
        mask = scipy.io.loadmat(video_path[:-4] + "_segm.mat", squeeze_me=True)
        mask = mask["segm_1"] > 0
        frame = frame * mask[:, :, None]
        return frame, mask
    else:
        return frame, None


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
    frame, mask = read_frame(path, SEGMENTATION)
    x1, x2, y1, y2, A_new, resized_K = read_pose_and_crop(path)
    if x1 is None:
        return None, None, None
    cropped_frame = frame[y1:y2, x1:x2]
    resized_frame = cv2.resize(cropped_frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    if SEGMENTATION:
        cropped_mask = mask[y1:y2, x1:x2].astype("uint8")
        resized_mask = cv2.resize(cropped_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        resized_frame = np.concatenate([resized_mask[:, :, None], resized_frame], axis=-1)

    return resized_frame, resized_K, A_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    # DATA_ROOT = "/data/unagi0/noguchi/dataset/SURREAL/SURREAL/data/cmu/"
    DATA_ROOT = args.data_path
    SEGMENTATION = True

    video_path = glob.glob(f"{DATA_ROOT}/*/*/*/*.mp4")
    print(len(video_path))

    img_cache = []
    pose_cache = []
    intrinsic_cache = []
    for path in tqdm(video_path):
        img, K, pose_3d = preprocess(path)
        if img is not None:
            img_cache.append(blosc.pack_array(img[:, :, ::-1].transpose(2, 0, 1)))
            pose_cache.append(pose_3d[0])
            intrinsic_cache.append(K)
        else:
            print("invalid data")

    img_cache = np.array(img_cache, dtype="object")
    pose_cache = np.array(pose_cache)
    intrinsic_cache = np.array(intrinsic_cache)
    cache = {"img": img_cache,
             "camera_intrinsic": intrinsic_cache,
             "smpl_pose": pose_cache}

    if SEGMENTATION:
        cache_dir_name = "VAE_cache"
    else:
        cache_dir_name = "GAN_cache"
    os.makedirs(f"{DATA_ROOT}/{cache_dir_name}", exist_ok=True)
    with open(f"{DATA_ROOT}/{cache_dir_name}/cache.pickle", "wb") as f:
        pickle.dump(cache, f)

    np.save(f"{DATA_ROOT}/{cache_dir_name}/canonical.npy",
            np.load("../smpl_data/neutral_canonical.npy"))
