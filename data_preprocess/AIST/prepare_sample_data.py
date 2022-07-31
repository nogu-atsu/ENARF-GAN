import argparse
import glob
import math
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from aist_plusplus.loader import AISTDataset
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../")
from libraries.smpl_utils import get_pose
from data_preprocess.utils import get_bone_length, SMPL_PARENTS


def preprocess(intrinsic, rot, trans, pose):
    if algo == "resize":
        intri = intrinsic.copy()
        intri[:2] /= save_scale
        return intri
    elif algo == "aligned_crop":
        focal_length = (intrinsic[0, 0] + intrinsic[1, 1]) / 2
        _crop_size = int(crop_size * focal_length / standard_focal_length)
        _crop_size = _crop_size // 2 * 2
        joint_translation = pose[:, :3, 3:]  # (n_parts, 3, 1)
        pose_3d = np.matmul(rot[None], joint_translation) + trans[None]  # (n_parts, 3, 1)
        pose_2d = np.matmul(intrinsic[None], pose_3d)
        pose_2d = pose_2d[:, :2, 0] / pose_2d[:, 2:, 0]  # (n_parts, 2)

        spine = pose_2d[0]  # (B, 2)
        x1 = math.floor(spine[0]) - _crop_size // 2
        y1 = math.floor(spine[1]) - _crop_size // 2

        # camera intrinsic
        intri = intrinsic.copy()
        intri[:2, 2] -= np.array([x1, y1])
        intri[:2] /= (_crop_size / save_size)
        return intri

    else:
        raise ValueError()


def save_sample_data(person_id):
    all_video_path = sorted(glob.glob(f"{VIDEO_DIR}/*_d{person_id:0>2}_*.mp4"))
    print(len(all_video_path))

    sample_data = []

    for video_path in tqdm(all_video_path[:5]):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        seq_name, view = AISTDataset.get_seq_name(video_name)
        view_idx = AISTDataset.VIEWS.index(view)
        env_name = aist_dataset.mapping_seq2env[seq_name]
        cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)

        camera_mat = cgroup.cameras[view_idx].matrix
        rmat = cv2.Rodrigues(cgroup.cameras[view_idx].rvec)[0]
        tvec = cgroup.cameras[view_idx].tvec[:, None]

        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(aist_dataset.motion_dir, seq_name)
        poses = smpl_poses.reshape(-1, 24, 3)

        with torch.no_grad():
            smpl_pose = get_pose(smpl, body_pose=torch.tensor(poses[:1, 1:]).float(),
                                 global_orient=torch.tensor(poses[:1, 0:1, :]).float())
            smpl_pose = smpl_pose.numpy()
            smpl_pose[:, :, :3, 3] *= smpl_scaling
            smpl_pose[:, :, :3, 3] += smpl_trans[:1, None]

        intri = preprocess(camera_mat, rmat, tvec, smpl_pose[0])

        # rotate
        smpl_pose[:, :, :3, :3] = np.matmul(rmat, smpl_pose[:, :, :3, :3])
        smpl_pose[:, :, :3, 3:] = (np.matmul(rmat, smpl_pose[:, :, :3, 3:]) + tvec) / 100

        sample_data.append({
            "pose_to_camera": smpl_pose[0],
            "intrinsics": intri,
            "bone_length": get_bone_length(smpl_pose[0], SMPL_PARENTS)
        })

    out_dir = f"../data/aist++"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/sample_data.pickle", "wb") as f:
        pickle.dump(sample_data, f)

    np.save(f'{out_dir}/canonical.npy', np.load(f"{SMPL_MODEL_PATH}/male_canonical.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--annotation_path", type=str)
    args = parser.parse_args()

    algo = "aligned_crop"
    crop_size = 600
    save_size = 128
    standard_focal_length = 1500

    save_scale = crop_size / save_size

    # VIDEO_DIR = "/data/unagi0/noguchi/dataset/aist++"
    # ANNOTATION_DIR = "/data/unagi0/noguchi/dataset/aist++/aist_plusplus_final"
    VIDEO_DIR = args.data_path
    ANNOTATION_DIR = args.annotation_path

    SMPL_MODEL_PATH = "../smpl_data"
    smpl = SMPL(model_path=SMPL_MODEL_PATH, gender='MALE', batch_size=1)
    aist_dataset = AISTDataset(ANNOTATION_DIR)
    PERSON_ID = 1

    save_sample_data(PERSON_ID)
