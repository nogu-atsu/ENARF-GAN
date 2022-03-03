import glob
import os
import pickle

import blosc
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

from preprocess import IMG_SIZE, DATA_ROOT, read_pose_and_crop


def read_frame(video_path):
    depth = scipy.io.loadmat(video_path[:-4] + "_depth.mat", squeeze_me=True)
    depth = depth["depth_1"] > 0  # (240, 320)
    disparity = 1 / depth
    disparity[disparity < 0.1] = 0
    return disparity


def preprocess(path):
    disparity = read_frame(path)

    x1, x2, y1, y2, A_new, resized_K = read_pose_and_crop(path)
    cropped_disparity = disparity[y1:y2, x1:x2]
    resized_disparity = cv2.resize(cropped_disparity, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    return resized_disparity, resized_K, A_new


if __name__ == "__main__":
    video_path = glob.glob(f"{DATA_ROOT}/*/*/*/*.mp4")
    print(len(video_path))

    disparity_cache = []
    pose_cache = []
    intrinsic_cache = []
    for path in tqdm(video_path):
        disparity, K, pose_3d = preprocess(path)
        if disparity is not None:
            disparity_cache.append(blosc.pack_array(disparity))
            pose_cache.append(pose_3d[0])
            intrinsic_cache.append(K)
        else:
            print("invalid data")

    disparity_cache = np.array(disparity_cache, dtype="object")
    pose_cache = np.array(pose_cache)
    intrinsic_cache = np.array(intrinsic_cache)
    cache = {"disparity": disparity_cache,
             "camera_intrinsic": intrinsic_cache,
             "smpl_pose": pose_cache}

    cache_dir_name = "NARF_GAN_depth_cache"
    os.makedirs(f"{DATA_ROOT}/{cache_dir_name}", exist_ok=True)
    with open(f"{DATA_ROOT}/{cache_dir_name}/cache.pickle", "wb") as f:
        pickle.dump(cache, f)

    np.save(f"{DATA_ROOT}/{cache_dir_name}/canonical.npy",
            np.load("../../smpl_data/neutral_canonical.npy"))
