import argparse
import json
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.append("../")
from data_preprocess.utils import get_bone_length, SMPL_PARENTS



def save_samples(n_frame, interval=1):
    split = "testing"
    camera_id = 0
    intrinsics = np.loadtxt(f"{DIR_PATH}/{PERSON_NAME}/intrinsic/0_train_{camera_id:0>4}.txt")
    extrinsics = np.linalg.inv(np.loadtxt(f"{DIR_PATH}/{PERSON_NAME}/pose/0_train_{camera_id:0>4}.txt"))
    sample_data = []
    for frame_id in tqdm(range(0, n_frame * interval, interval)):
        motion_path = f"{DIR_PATH}/{PERSON_NAME}/{split}/transform_smoth3e-2_withmotion/{frame_id:0>6}.json"
        with open(motion_path) as f:
            data = json.load(f)
        joints_RT = np.array(data["joints_RT"])
        rotation = np.array(data["rotation"])
        joints = np.array(data["joints"])
        joint_rot = np.matmul(rotation.T, joints_RT.transpose(2, 0, 1)[:, :3, :3])
        joint_transform = np.concatenate([joint_rot, joints[:, :, None]], axis=-1)  # (24, 3, 4)
        joint_transform = np.concatenate([joint_transform, np.tile(np.array([0, 0, 0, 1])[None, None], (24, 1, 1))],
                                         axis=1)
        pose_3d = np.matmul(extrinsics, joint_transform)

        sample_data.append({"pose_3d": pose_3d,
                            "intrinsics": intrinsics,
                            "bone_length": get_bone_length(pose_3d, SMPL_PARENTS)})
    out_dir = f"../data/NeuralActor/{PERSON_NAME}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/sample_data.pickle", "wb") as f:
        pickle.dump(sample_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--person_name", type=str, default="lan", help="person name. 'lan' or 'marc'")
    args = parser.parse_args()

    DIR_PATH = args.data_path
    PERSON_NAME = args.person_name
    IMAGE_SIZE = 1024

    save_samples(n_frame=5, interval=1000)
