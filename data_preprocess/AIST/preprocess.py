import argparse
import glob
import math
import os
import pickle
import sys
from contextlib import redirect_stdout

import blosc
import cv2
import numpy as np
import torch
from aist_plusplus import utils
from aist_plusplus.loader import AISTDataset
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../../")
from dependencies.smpl_utils import get_pose


def preprocess_imgs(img, intrinsic, rot, trans, pose):
    if algo == "resize":
        img = img[:crop_size, :crop_size]
        img = cv2.resize(img, (save_size, save_size), interpolation=cv2.INTER_CUBIC)
        img = img[:, :, ::-1]
        intri = intrinsic.copy()
        intri[:2] /= save_scale
        return img, intri, True
    elif algo == "aligned_crop":
        focal_length = (intrinsic[0, 0] + intrinsic[1, 1]) / 2
        _crop_size = int(crop_size * focal_length / standard_focal_length)
        _crop_size = _crop_size // 2 * 2
        joint_translation = pose[:, :3, 3:]  # (n_parts, 3, 1)
        pose_3d = np.matmul(rot[None], joint_translation) + trans[None]  # (n_parts, 3, 1)
        pose_2d = np.matmul(intrinsic[None], pose_3d)
        pose_2d = pose_2d[:, :2, 0] / pose_2d[:, 2:, 0]  # (n_parts, 2)

        # check if all joints are in original img
        h, w, _ = img.shape
        validity = pose_2d.min() >= 0 and pose_2d[:, 0].max() < w and pose_2d[:, 1].max() < h

        if validity:
            spine = pose_2d[0]  # (B, 2)
            x1 = math.floor(spine[0]) - _crop_size // 2
            x2 = math.floor(spine[0]) + _crop_size // 2
            y1 = math.floor(spine[1]) - _crop_size // 2
            y2 = math.floor(spine[1]) + _crop_size // 2
            img = np.pad(img, ((max(0, -y1), max(0, y2 - h)),
                               (max(0, -x1), max(0, x2 - w)),
                               (0, 0)), mode="reflect")
            img = img[max(0, y1):max(0, y1) + _crop_size, max(0, x1):max(0, x1) + _crop_size]
            img = cv2.resize(img, (save_size, save_size), interpolation=cv2.INTER_CUBIC)
            img = img[:, :, ::-1]
        else:
            print("invalid")
            img = np.zeros((save_size, save_size, 3), dtype="uint8")
            x1, y1 = 0, 0

        # camera intrinsic
        intri = intrinsic.copy()
        intri[:2, 2] -= np.array([x1, y1])
        intri[:2] /= (_crop_size / save_size)
        return img, intri, validity

    else:
        raise ValueError()


def read_frames(person_id):
    all_video_path = sorted(glob.glob(f"{VIDEO_DIR}/*_d{person_id:0>2}_*.mp4"))
    print(len(all_video_path))

    all_video = []
    all_processed_intrinsic = []
    all_validity = []
    all_rmat = []
    all_tvec = []
    all_smpl = []

    for video_path in tqdm(all_video_path):
        print(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        seq_name, view = AISTDataset.get_seq_name(video_name)
        view_idx = AISTDataset.VIEWS.index(view)
        env_name = aist_dataset.mapping_seq2env[seq_name]
        cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)
        # # no print
        with redirect_stdout(open(os.devnull, 'w')):
            frames3fps = utils.ffmpeg_video_read(video_path, 3)[:, :, :, ::-1]  # BGR

        camera_mat = cgroup.cameras[view_idx].matrix
        rmat = cv2.Rodrigues(cgroup.cameras[view_idx].rvec)[0]
        tvec = cgroup.cameras[view_idx].tvec[:, None]
        dist = cgroup.cameras[view_idx].dist

        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(aist_dataset.motion_dir, seq_name)
        poses = smpl_poses.reshape(-1, 24, 3)

        frames = []
        processed_intrinsic = []
        validities = []

        with torch.no_grad():
            all_smpl_param = get_pose(smpl, body_pose=torch.tensor(poses[:, 1:]).float(),
                                      global_orient=torch.tensor(poses[:, 0:1, :]).float())
            all_smpl_param = all_smpl_param.numpy()
            all_smpl_param[:, :, :3, 3] *= smpl_scaling
            all_smpl_param[:, :, :3, 3] += smpl_trans[:, None]

        video_len = len(frames3fps)
        all_rmat.append(np.tile(rmat, (video_len, 1, 1)))
        all_tvec.append(np.tile(tvec, (video_len, 1, 1)))
        all_smpl.append(all_smpl_param[19:20 * video_len:20])

        for i in range(video_len):
            # preprocess images
            idx_60fps = (i + 1) * 20 - 1
            frame = cv2.undistort(frames3fps[i], camera_mat, dist)
            frame, intri, validity = preprocess_imgs(frame, camera_mat, rmat,
                                                     tvec, all_smpl_param[idx_60fps])

            frames.append(frame)
            processed_intrinsic.append(intri)
            validities.append(validity)

        all_video.append(np.array(frames))
        all_processed_intrinsic.append(np.array(processed_intrinsic))
        all_validity.append(np.array(validities))

    all_video = np.concatenate(all_video, axis=0)
    all_processed_intrinsic = np.concatenate(all_processed_intrinsic, axis=0)
    all_validity = np.concatenate(all_validity, axis=0)
    all_rmat = np.concatenate(all_rmat, axis=0)
    all_tvec = np.concatenate(all_tvec, axis=0)
    all_smpl = np.concatenate(all_smpl, axis=0)

    # discard invalid
    all_video = all_video[all_validity]
    all_processed_intrinsic = all_processed_intrinsic[all_validity]
    all_rmat = all_rmat[all_validity]
    all_tvec = all_tvec[all_validity]
    all_smpl = all_smpl[all_validity]

    print("num frames", len(all_video))
    sample_idx = np.linspace(0, len(all_video) - 1, N_PER_PERSON, dtype="int")

    all_video = all_video[sample_idx]
    all_processed_intrinsic = all_processed_intrinsic[sample_idx]
    all_rmat = all_rmat[sample_idx]
    all_tvec = all_tvec[sample_idx].copy()
    all_smpl = all_smpl[sample_idx].copy()

    # normalize
    all_tvec = all_tvec / 100
    all_smpl[:, :, :3, 3] /= 100

    return all_video, all_rmat, all_tvec, all_smpl, all_processed_intrinsic


def create_dict(video, all_rot, all_trans, smpl, all_intrinsic):
    data_dict = {}
    data_dict["img"] = np.array([blosc.pack_array(frame.transpose(2, 0, 1)) for frame in tqdm(video)],
                                dtype="object")

    data_dict["camera_intrinsic"] = all_intrinsic
    data_dict["camera_rotation"] = all_rot
    data_dict["camera_translation"] = all_trans

    data_dict["smpl_pose"] = smpl
    return data_dict


def save_cache(person_ids: np.ndarray) -> None:
    for person_id in person_ids:
        # read frame
        outputs = read_frames(person_id)

        train_dict = create_dict(*outputs)

        cache_path = f'{VIDEO_DIR}/cache{save_size}_{algo}_fl{standard_focal_length}/{person_id:0>2}'
        os.makedirs(cache_path, exist_ok=True)
        with open(f'{cache_path}/cache.pickle', 'wb') as f:
            pickle.dump(train_dict, f)


def merge_all_cache(person_ids, mode="train"):
    cache_path = f'{VIDEO_DIR}/cache{save_size}_{algo}_fl{standard_focal_length}'
    all_data = []
    all_data_dict = {}
    for person_id in person_ids:
        with open(f'{cache_path}/{person_id:0>2}/cache.pickle', 'rb') as f:
            all_data.append(pickle.load(f))

    for k in all_data[0]:
        all_data_dict[k] = np.concatenate([data[k] for data in all_data], axis=0)

    os.makedirs(f'{cache_path}/all_{mode}', exist_ok=True)
    with open(f'{cache_path}/all_{mode}/cache.pickle', 'wb') as f:
        pickle.dump(all_data_dict, f)

    np.save(f'{cache_path}/all_{mode}/canonical.npy', np.load(f"{SMPL_MODEL_PATH}/male_canonical.npy"))


def preprocess():
    save_cache(train_person_ids)
    save_cache(test_person_ids)

    merge_all_cache(np.concatenate([train_person_ids,
                                    test_person_ids]), "all")


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
    SMPL_MODEL_PATH = "../../smpl_data"
    smpl = SMPL(model_path=SMPL_MODEL_PATH, gender='MALE', batch_size=1)
    aist_dataset = AISTDataset(ANNOTATION_DIR)
    train_person_ids = np.arange(7, 31)
    test_person_ids = np.arange(1, 7)
    N_PER_PERSON = 3000

    preprocess()
