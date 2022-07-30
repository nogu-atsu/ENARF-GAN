import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../")
from dependencies.smpl_utils import get_pose
from data_preprocess.utils import get_bone_length, SMPL_PARENTS
from data_preprocess.surreal.preprocess import read_annots, IMG_SIZE, CROP_SIZE, read_pose_and_crop

SMPL_MODEL = {"male": SMPL(model_path="../smpl_data", gender="male"),
              "female": SMPL(model_path="../smpl_data", gender="female")}


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
    SMPL_MODEL_PATH = "../smpl_data"

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
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/sample_data.pickle", "wb") as f:
        pickle.dump(sample_data, f)

    np.save(f'{out_dir}/canonical.npy', np.load(f"{SMPL_MODEL_PATH}/neutral_canonical.npy"))
