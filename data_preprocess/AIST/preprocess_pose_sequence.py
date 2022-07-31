# create pose sequence
import glob
import math
import os
import pickle
import random
import sys

import cv2
import numpy as np
import torch
from aist_plusplus.loader import AISTDataset
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../../")
from libraries.smpl_utils import get_pose


def preprocess_intrinsic(intrinsic, rot, trans, pose):
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

        # check if all joints are in original img
        # validity = pose_2d.min() >= 0 and pose_2d[:, 0].max() < w and pose_2d[:, 1].max() < h
        validity = True

        if validity:
            spine = pose_2d[0]  # (B, 2)
            x1 = math.floor(spine[0]) - _crop_size // 2
            y1 = math.floor(spine[1]) - _crop_size // 2
        else:
            print("invalid")
            x1, y1 = 0, 0

        # camera intrinsic
        intri = intrinsic.copy()
        intri[:2, 2] -= np.array([x1, y1])
        intri[:2] /= (_crop_size / save_size)
        return intri

    else:
        raise ValueError()



def read_pose_sequences(person_id):
    all_video_path = sorted(glob.glob(f"{VIDEO_DIR}/*_d{person_id:0>2}_*.mp4"))
    print(len(all_video_path))

    video_path = random.choice(all_video_path)
    print(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    seq_name, view = AISTDataset.get_seq_name(video_name)
    view_idx = AISTDataset.VIEWS.index(view)
    env_name = aist_dataset.mapping_seq2env[seq_name]
    cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)
    # no print
    # with redirect_stdout(open(os.devnull, 'w')):
    #     frames3fps = utils.ffmpeg_video_read(video_path, 3)[:, :, :, ::-1]  # BGR

    camera_mat = cgroup.cameras[view_idx].matrix
    rmat = cv2.Rodrigues(cgroup.cameras[view_idx].rvec)[0]
    tvec = cgroup.cameras[view_idx].tvec[:, None]

    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(aist_dataset.motion_dir, seq_name)
    poses = smpl_poses.reshape(-1, 24, 3)

    processed_intrinsic = []

    with torch.no_grad():
        all_smpl_param = get_pose(smpl, body_pose=torch.tensor(poses[:, 1:]).float(),
                                  global_orient=torch.tensor(poses[:, 0:1, :]).float())
        all_smpl_param = all_smpl_param.numpy()
        all_smpl_param[:, :, :3, 3] *= smpl_scaling
        all_smpl_param[:, :, :3, 3] += smpl_trans[:, None]

    video_len = len(all_smpl_param)
    all_rmat = np.tile(rmat, (video_len, 1, 1))
    all_tvec = np.tile(tvec, (video_len, 1, 1))
    all_smpl = all_smpl_param

    for i in range(video_len):
        intri = preprocess_intrinsic(camera_mat, rmat,
                                     tvec, all_smpl_param[i])

        processed_intrinsic.append(intri)

    all_processed_intrinsic = np.array(processed_intrinsic)

    print("num frames", len(all_smpl))

    all_tvec = all_tvec.copy()
    all_smpl = all_smpl.copy()

    # normalize
    all_tvec = all_tvec / 100
    all_smpl[:, :, :3, 3] /= 100

    return all_rmat, all_tvec, all_smpl, all_processed_intrinsic


def create_dict(all_rot, all_trans, smpl, all_intrinsic):
    data_dict = {}

    data_dict["camera_intrinsic"] = all_intrinsic
    data_dict["camera_rotation"] = all_rot
    data_dict["camera_translation"] = all_trans

    data_dict["smpl_pose"] = smpl
    return data_dict


def save_cache(person_ids: np.ndarray) -> None:
    for person_id in tqdm(person_ids):
        # read frame
        outputs = read_pose_sequences(person_id)

        train_dict = create_dict(*outputs)

        cache_path = f'{VIDEO_DIR}/sequence_cache/{person_id:0>2}'
        os.makedirs(cache_path, exist_ok=True)
        with open(f'{cache_path}/cache.pickle', 'wb') as f:
            pickle.dump(train_dict, f)

        np.save(f'{cache_path}/canonical.npy', np.load(f"{SMPL_MODEL_PATH}/male_canonical.npy"))


def preprocess():
    save_cache(np.concatenate([train_person_ids,
                               test_person_ids]))


if __name__ == "__main__":
    # algo = "resize"  # resize, aligned_crop
    # crop_size =
    # save_size =
    # standard_focal_length = None

    algo = "aligned_crop"
    crop_size = 600
    save_size = 128
    standard_focal_length = 1500

    save_scale = crop_size / save_size

    VIDEO_DIR = "/data/unagi0/noguchi/dataset/aist++"
    ANNOTATION_DIR = "/data/unagi0/noguchi/dataset/aist++/aist_plusplus_final"
    SMPL_MODEL_PATH = "../../smpl_data"
    smpl = SMPL(model_path=SMPL_MODEL_PATH, gender='MALE', batch_size=1)
    aist_dataset = AISTDataset(ANNOTATION_DIR)
    train_person_ids = np.arange(7, 31)
    test_person_ids = np.arange(1, 7)

    preprocess()
