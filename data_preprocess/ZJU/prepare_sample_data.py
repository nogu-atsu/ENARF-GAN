import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../")
from libraries.smpl_utils import get_pose
from data_preprocess.utils import get_bone_length, SMPL_PARENTS


def read_annots(person_id):
    annot = np.load(f"{DIR_PATH}/CoreView_{person_id}/annots.npy", allow_pickle=True)
    K = np.array(annot[()]['cams']['K'])
    R = np.array(annot[()]['cams']['R'])
    T = np.array(annot[()]['cams']['T']) / 1000
    D = np.array(annot[()]['cams']['D'])
    n_camera = len(K)
    cam_trans = np.broadcast_to(np.eye(4), (n_camera, 4, 4)).copy()
    cam_trans[:, :3, :3] = R
    cam_trans[:, :3, 3] = np.array(T).squeeze(-1)
    image_paths = annot[()]["ims"]
    return K, R, T, D, cam_trans, image_paths, n_camera


def save_samples(n_frame, K, R, T, smpl, start_frame_idx=0, interval=1):
    sample_data = []

    # resize
    K_new = K[CAMERA_ID].copy()
    K_new[:2] *= IMAGE_SIZE / ORG_IMAGE_SIZE

    # compute extrinsic matrix from R and T
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R[CAMERA_ID]
    extrinsics[:3, 3] = T[CAMERA_ID, :, 0]

    for frame_id in tqdm(range(start_frame_idx, start_frame_idx + n_frame * interval, interval)):
        smpl_idx = frame_id + 1 if PERSON_ID in ["313", "315"] else frame_id
        smpl_param = np.load(f"{DIR_PATH}/CoreView_{PERSON_ID}/new_params/{smpl_idx}.npy", allow_pickle=True)
        poses = smpl_param[()]['poses'].reshape(1, 24, 3)
        shapes = smpl_param[()]['shapes']

        trans = np.eye(4)
        trans[:3, :3] = cv2.Rodrigues(smpl_param[()]['Rh'])[0]
        trans[:3, 3] = smpl_param[()]['Th']

        with torch.no_grad():
            pose = get_pose(smpl, torch.tensor(shapes), body_pose=torch.tensor(poses[:, 1:]).float(),
                            global_orient=torch.tensor(poses[:, 0:1, :]).float()).numpy()[0]
            pose_to_world = np.matmul(trans, pose)
            pose_3d = np.matmul(extrinsics, pose_to_world)

        sample_data.append({"pose_3d": pose_3d,
                            "intrinsics": K_new,
                            "bone_length": get_bone_length(pose_3d, SMPL_PARENTS)})

    out_dir = f"../data/ZJU_DSO/{PERSON_ID}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/sample_data.pickle", "wb") as f:
        pickle.dump(sample_data, f)


def preprocess():
    smpl = SMPL(model_path=SMPL_MODEL_PATH, gender='NEUTRAL', batch_size=1)
    K, R, T, D, cam_trans, image_paths, n_camera = read_annots(PERSON_ID)
    save_samples(5, K, R, T, smpl, interval=30, start_frame_idx=n_train_frames[PERSON_ID])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--person_id", type=str, default="lan", help="person name. 'lan' or 'marc'")
    args = parser.parse_args()

    SMPL_MODEL_PATH = "../smpl_data"
    DIR_PATH = args.data_path
    IMAGE_SIZE = 512
    ORG_IMAGE_SIZE = 1024  # need check
    CAMERA_ID = 1
    PERSON_ID = args.person_id

    n_train_frames = {"313": 1176, "315": 1748, "386": 516}

    preprocess()
