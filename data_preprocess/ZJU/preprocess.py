# this code is based on the code of another ongoing project
import json
import os
import pickle
from typing import List
import glob
import math

import blosc
import cv2
import numpy as np
from easymocap.smplmodel import SMPLlayer
from tqdm import tqdm

from read_smpl import PoseLoader


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
        spine = pose_2d[0]  # (B, 2)
        img_size = img.shape[0]
        x1 = math.floor(spine[0]) - _crop_size // 2
        x2 = math.floor(spine[0]) + _crop_size // 2
        y1 = math.floor(spine[1]) - _crop_size // 2
        y2 = math.floor(spine[1]) + _crop_size // 2
        img = np.pad(img, ((max(0, -x1), max(0, x2 - img_size)),
                           (max(0, -y1), max(0, y2 - img_size)),
                           (0, 0)), mode="reflect")
        img = img[max(0, y1):max(0, y1) + _crop_size, max(0, x1):max(0, x1) + _crop_size]
        img = cv2.resize(img, (save_size, save_size), interpolation=cv2.INTER_CUBIC)
        img = img[:, :, ::-1]

        # check if all joints are in original img
        validity = pose_2d.min() >= 0 and pose_2d.max() < img_size

        # camera intrinsic
        intri = intrinsic.copy()
        intri[:2, 2] -= np.array([x1, y1])
        intri[:2] /= (_crop_size / save_size)
        return img, intri, validity

    else:
        raise ValueError()


def read_frames(person_id, chosen_camera_id, all_intrinsic, all_rot, all_trans, all_smpl_param):
    all_video = []
    all_processed_intrinsic = []
    camera_id = []
    all_validity = []
    for cam in tqdm(chosen_camera_id):
        video_path = f"{ZJU_PATH}/{person_id}/videos/{cam:0>2}.mp4"
        video = cv2.VideoCapture(video_path)
        frames = []
        processed_intrinsic = []
        validities = []
        frame_id = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # preprocess images
            frame, intri, validity = preprocess_imgs(frame, all_intrinsic[cam - 1], all_rot[cam - 1],
                                                     all_trans[cam - 1], all_smpl_param[frame_id])

            frames.append(frame)
            processed_intrinsic.append(intri)
            validities.append(validity)
            frame_id += 1

        all_video.append(np.array(frames))
        all_processed_intrinsic.append(np.array(processed_intrinsic))
        all_validity.append(np.array(validities))
        camera_id.append(np.ones(len(frames), dtype="int") * (cam - 1))

    video_len = np.array([video.shape[0] for video in all_video])
    assert (video_len == video_len[0]).all()
    video_len = video_len[0]
    frame_id = [np.arange(video_len) for _ in range(num_camera)]

    # thin out
    def thin_out_and_cat(arrays):
        arrays = [arr[i % thin_out_rate:video_len // thin_out_rate * thin_out_rate:thin_out_rate] for i, arr in
                  enumerate(arrays)]
        return np.concatenate(arrays, axis=0)

    all_video = thin_out_and_cat(all_video)
    frame_id = thin_out_and_cat(frame_id)
    camera_id = thin_out_and_cat(camera_id)
    all_validity = thin_out_and_cat(all_validity)
    all_processed_intrinsic = thin_out_and_cat(all_processed_intrinsic)

    assert all_video.shape[0] == video_len // thin_out_rate * num_camera

    return all_video, frame_id, camera_id, video_len // thin_out_rate, all_processed_intrinsic, all_validity


def read_intrinsic(person_id):
    fs = cv2.FileStorage(f"{ZJU_PATH}/{person_id}/intri.yml", cv2.FILE_STORAGE_READ)
    all_intrinsic = []
    for cam in range(1, num_camera + 1):
        matrix = fs.getNode(f"K_{cam:0>2}").mat()
        matrix = np.array(matrix).reshape(3, 3)
        all_intrinsic.append(matrix)
    all_intrinsic = np.array(all_intrinsic)
    return all_intrinsic


def read_extrinsic(person_id):
    fs = cv2.FileStorage(f"{ZJU_PATH}/{person_id}/extri.yml", cv2.FILE_STORAGE_READ)
    all_rot = []
    all_trans = []
    for cam in range(1, num_camera + 1):
        rot = fs.getNode(f"Rot_{cam:0>2}").mat()
        rot = np.array(rot).reshape(3, 3)
        trans = fs.getNode(f"T_{cam:0>2}").mat()
        trans = np.array(trans).reshape(3, 1)
        all_rot.append(rot)
        all_trans.append(trans)
    all_rot = np.array(all_rot)
    all_trans = np.array(all_trans)
    return all_rot, all_trans


def read_smpl_parameters(person_id):
    all_smpl_param = []
    smpl_paths = sorted(glob.glob(f"{ZJU_PATH}/{person_id}/smplx/*.json"))
    for smpl_path in smpl_paths:
        with open(smpl_path, "r") as f:
            smpl_param = json.load(f)[0]

        smpl_param = pose_loader(smpl_param)
        all_smpl_param.append(smpl_param)
    all_smpl_param = np.array(all_smpl_param)
    return all_smpl_param


def create_dict(video, frame, camera, all_intrinsic, all_rot, all_trans, smpl, validity, set_size):
    data_dict = {}
    data_dict["frame_id"] = frame.reshape(-1)
    data_dict["img"] = np.array([blosc.pack_array(frame.transpose(2, 0, 1)) for frame in tqdm(video)],
                                dtype="object")

    data_dict["camera_intrinsic"] = all_intrinsic
    data_dict["camera_rotation"] = all_rot[camera]
    data_dict["camera_translation"] = all_trans[camera]

    data_dict["camera_id"] = np.arange(len(frame)) // set_size
    data_dict["smpl_pose"] = smpl[frame]
    data_dict["validity"] = validity
    return data_dict


def save_cache(person_ids: List[int]) -> None:
    for person_id in person_ids:
        # read data other than imgs
        all_smpl_param = read_smpl_parameters(person_id)
        all_intrinsic = read_intrinsic(person_id)
        all_rot, all_trans = read_extrinsic(person_id)

        # read frame
        (all_video, frame_id, camera_id, video_len,
         all_intrinsic, all_validity) = read_frames(person_id, all_camera_id, all_intrinsic,
                                                    all_rot, all_trans, all_smpl_param)

        train_dict = create_dict(all_video, frame_id, camera_id, all_intrinsic,
                                 all_rot, all_trans, all_smpl_param, all_validity, video_len)

        os.makedirs(
            f'{ZJU_PATH}/cache{save_size}_{algo}_fl{standard_focal_length}_correct_{sampling_rate}%/{person_id}',
            exist_ok=True)
        with open(
                f'{ZJU_PATH}/cache{save_size}_{algo}_fl{standard_focal_length}_correct_{sampling_rate}%/{person_id}/cache.pickle',
                'wb') as f:
            pickle.dump(train_dict, f)


def merge_all_cache(person_ids, mode="train"):
    all_data = []
    all_data_dict = {}
    for person_id in person_ids:
        with open(
                f'{ZJU_PATH}/cache{save_size}_{algo}_fl{standard_focal_length}_correct_{sampling_rate}%/{person_id}/cache.pickle',
                'rb') as f:
            all_data.append(pickle.load(f))

    validity = np.concatenate([data["validity"] for data in all_data], axis=0)

    for k in all_data[0]:
        all_data_dict[k] = np.concatenate([data[k] for data in all_data], axis=0)[validity]

    os.makedirs(f'{ZJU_PATH}/cache{save_size}_{algo}_fl{standard_focal_length}_correct_{sampling_rate}%/all_{mode}',
                exist_ok=True)
    with open(
            f'{ZJU_PATH}/cache{save_size}_{algo}_fl{standard_focal_length}_correct_{sampling_rate}%/all_{mode}/cache.pickle',
            'wb') as f:
        pickle.dump(all_data_dict, f)


def preprocess():
    save_cache(train_person_ids)
    save_cache(test_person_ids)

    merge_all_cache(train_person_ids, "train")
    merge_all_cache(test_person_ids, "test")


if __name__ == "__main__":
    # algo = "resize"  # resize, aligned_crop
    # crop_size = 1024
    # save_size = 256
    # standard_focal_length = None

    algo = "aligned_crop"
    crop_size = 600
    save_size = 128
    standard_focal_length = 1200

    num_camera = 23
    save_scale = crop_size / save_size
    thin_out_rate = 5
    sampling_rate = 100 // thin_out_rate

    SMPL_MODEL_PATH = "/home/mil/noguchi/D2/EasyMocap/data/smplx"
    ZJU_PATH = "/data/unagi0/noguchi/dataset/zju_mocap_v2"

    pose_loader = PoseLoader(SMPL_MODEL_PATH)
    smpllayer = SMPLlayer(SMPL_MODEL_PATH + "/smplx", model_type='smplx', use_joints=False)

    all_camera_id = np.arange(1, num_camera + 1)

    train_person_ids = [363, 364, 365, 366, 367, 368, 369, 370, 371, 377, 378, 379, 380, 381, 382, 383]
    test_person_ids = [384, 385, 386, 387]

    preprocess()
