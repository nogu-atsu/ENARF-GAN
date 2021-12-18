# this code is based on the code of another ongoing project
import json
import os
import pickle
from typing import List

import blosc
import cv2
import numpy as np
from easymocap.smplmodel import SMPLlayer
from tqdm import tqdm

from read_smpl import PoseLoader


def read_frames(person_id, save_size, crop_size, chosen_camera_id):
    all_video = []
    for cam in tqdm(chosen_camera_id):
        video_path = f"{ZJU_PATH}/{person_id}/videos/{cam:0>2}.mp4"
        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = frame[:crop_size, :crop_size]
            frame = cv2.resize(frame, (save_size, save_size), interpolation=cv2.INTER_CUBIC)
            frames.append(frame[:, :, ::-1])
        frames = np.array(frames)
        all_video.append(frames)
    video_len = np.array([video.shape[0] for video in all_video])
    assert (video_len == video_len[0]).all()
    video_len = video_len[0]
    frame_id = [np.arange(video_len) for _ in range(num_camera)]
    # thin out
    all_video = [video[i % thin_out_rate:video_len // thin_out_rate * thin_out_rate:thin_out_rate] for i, video in
                 enumerate(all_video)]
    frame_id = [frame[i % thin_out_rate:video_len // thin_out_rate * thin_out_rate:thin_out_rate] for i, frame in
                enumerate(frame_id)]

    frame_id = np.concatenate(frame_id, axis=0)
    all_video = np.concatenate(all_video, axis=0)
    assert all_video.shape[0] == video_len // thin_out_rate * num_camera

    camera_id = [np.ones(video_len // thin_out_rate, dtype="int") * (cam - 1) for cam in chosen_camera_id]
    camera_id = np.concatenate(camera_id, axis=0)
    return all_video, frame_id, camera_id, video_len // thin_out_rate


def read_intrinsic(person_id, save_scale):
    fs = cv2.FileStorage(f"{ZJU_PATH}/{person_id}/intri.yml", cv2.FILE_STORAGE_READ)
    all_intrinsic = []
    for cam in range(1, num_camera + 1):
        matrix = fs.getNode(f"K_{cam:0>2}").mat()
        matrix = np.array(matrix).reshape(3, 3)
        all_intrinsic.append(matrix)
    all_intrinsic = np.array(all_intrinsic)
    all_intrinsic[:, :2] /= save_scale
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


def read_smpl_parameters(person_id, video_len):
    all_smpl_param = []
    for frame_id in tqdm(range(video_len)):
        smpl_path = f"{ZJU_PATH}/{person_id}/smplx/{frame_id:0>6}.json"

        with open(smpl_path, "r") as f:
            smpl_param = json.load(f)[0]

        smpl_param = pose_loader(smpl_param)
        all_smpl_param.append(smpl_param)
    all_smpl_param = np.array(all_smpl_param)
    return all_smpl_param


def create_dict(video, frame, camera, all_intrinsic, all_rot, all_trans, smpl, set_size):
    data_dict = {}
    data_dict["frame_id"] = frame.reshape(-1)
    data_dict["img"] = np.array([blosc.pack_array(frame.transpose(2, 0, 1)) for frame in tqdm(video)],
                                dtype="object")

    data_dict["camera_intrinsic"] = all_intrinsic[camera]
    data_dict["camera_rotation"] = all_rot[camera]
    data_dict["camera_translation"] = all_trans[camera]

    data_dict["camera_id"] = np.arange(len(frame)) // set_size
    data_dict["smpl_pose"] = smpl
    return data_dict


def save_cache(person_ids: List[int]) -> None:
    for person_id in person_ids:
        # read frame
        all_video, frame_id, camera_id, video_len = read_frames(person_id, save_size, crop_size, all_camera_id)
        all_smpl_param = read_smpl_parameters(person_id, video_len)
        all_intrinsic = read_intrinsic(person_id, save_scale)
        all_rot, all_trans = read_extrinsic(person_id)

        train_dict = create_dict(all_video, frame_id, camera_id, all_intrinsic,
                                 all_rot, all_trans, all_smpl_param, video_len)

        os.makedirs(f'{ZJU_PATH}/cache{save_size}_correct_{sampling_rate}%/{person_id}', exist_ok=True)
        with open(f'{ZJU_PATH}/cache{save_size}_correct_{sampling_rate}%/{person_id}/cache.pickle', 'wb') as f:
            pickle.dump(train_dict, f)


def merge_all_cache(person_ids, mode="train"):
    all_data = []
    all_data_dict = {}
    for person_id in person_ids:
        with open(f'{ZJU_PATH}/cache{save_size}_correct_{sampling_rate}%/{person_id}/cache.pickle', 'rb') as f:
            all_data.append(pickle.load(f))

    for k in all_data[0]:
        all_data_dict[k] = np.concatenate([data[k] for data in all_data], axis=0)

    os.makedirs(f'{ZJU_PATH}/cache{save_size}_correct_{sampling_rate}%/all_{mode}', exist_ok=True)
    with open(f'{ZJU_PATH}/cache{save_size}_correct_{sampling_rate}%/all_{mode}/cache.pickle', 'wb') as f:
        pickle.dump(all_data_dict, f)


def preprocess():
    save_cache(train_person_ids)
    save_cache(test_person_ids)

    merge_all_cache(train_person_ids, "train")
    merge_all_cache(test_person_ids, "test")


if __name__ == "__main__":
    save_scale = 4
    crop_size = 1024
    num_camera = 23
    save_size = crop_size // save_scale
    thin_out_rate = 5
    sampling_rate = 100 // thin_out_rate

    SMPL_MODEL_PATH = "/home/mil/noguchi/D2/EasyMocap/data/smplx"
    ZJU_PATH = "/data/unagi0/noguchi/dataset/zju_mocap_v2"

    pose_loader = PoseLoader(SMPL_MODEL_PATH)
    smpllayer = SMPLlayer(SMPL_MODEL_PATH + "/smplx", model_type='smplx', use_joints=False)

    joints_to_use = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 37
    ])
    all_camera_id = np.arange(1, num_camera + 1)

    train_person_ids = [363, 364, 365, 366, 367, 368, 369, 370, 371, 377, 378, 379, 380, 381, 382, 383]
    test_person_ids = [384, 385, 386, 387]

    preprocess()
