import json
import os
import pickle

import blosc
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def get_mask_from_rgb(image):
    assert image.shape[0] == 3
    assert image.max() > 250
    mask = np.linalg.norm(image.astype("float") - 255, axis=0) >= 18
    return mask


class ReadSingleVideo:
    def __init__(self, n_camera, person_id, split, n_frame, interval):
        self.n_camera = n_camera
        self.person_id = person_id
        self.split = split
        self.n_frame = n_frame
        self.interval = interval

    def read_single_video(self, cam_id):
        count = 0
        frames = []
        cap = cv2.VideoCapture(f"{DIR_PATH}/{self.person_id}/{self.split}/rgb_video/{cam_id:0>3}.avi")
        pbar = tqdm(total=self.n_frame)
        while True:
            ret, frame = cap.read()
            if ret:
                if count % self.interval == 0:
                    frame = frame.transpose(2, 0, 1)[::-1]
                    mask = get_mask_from_rgb(frame)
                    frame = frame * mask + 255 * (1 - mask)
                    frame = frame.astype("uint8")
                    frames.append(blosc.pack_array(frame))
                count += 1
                pbar.update(1)
            else:
                pbar.close()
                break
        print("Video {} is done".format(cam_id))
        return frames


def read_frames(n_camera, person_id, split, n_frame, interval=1):
    rsv = ReadSingleVideo(n_camera, person_id, split, n_frame, interval)
    # # single process
    # frames = [rsv.read_single_video(pcam_id) for cam_id in range(n_camera)]

    # multi process
    p = Pool(5)
    frames = p.map(rsv.read_single_video, range(n_camera))

    np_frames = []
    for i in range(len(frames[0])):
        for cam_id in range(n_camera):
            np_frames.append(frames[cam_id][i])
    np_frames = np.array(np_frames, dtype="object")
    return np_frames


def save_cache(person_id, n_camera, n_frame, prefix="train", start_frame_idx=0, interval=1):
    if DEBUG:
        interval = 10
    split = "training" if prefix == "train" else "testing"
    np_frames = read_frames(n_camera, person_id, split, n_frame, interval)

    intrinsics = np.array(
        [np.loadtxt(f"{DIR_PATH}/{person_id}/intrinsic/0_train_{i:0>4}.txt") for i in range(n_camera)])
    extrinsics = np.array(
        [np.linalg.inv(np.loadtxt(f"{DIR_PATH}/{person_id}/pose/0_train_{i:0>4}.txt")) for i in range(n_camera)])
    smpl_pose = []
    for frame_id in tqdm(range(0, n_frame, interval)):
        motion_path = f"{DIR_PATH}/{person_id}/{split}/transform_smoth3e-2_withmotion/{frame_id:0>6}.json"
        with open(motion_path) as f:
            data = json.load(f)
        joints_RT = np.array(data["joints_RT"])
        translation = np.array(data["translation"])
        rotation = np.array(data["rotation"])
        joints = np.array(data["joints"])
        joint_rot = np.matmul(rotation.T, joints_RT.transpose(2, 0, 1)[:, :3, :3])
        joint_transform = np.concatenate([joint_rot, joints[:, :, None]], axis=-1)  # (24, 3, 4)
        joint_transform = np.concatenate([joint_transform, np.tile(np.array([0, 0, 0, 1])[None, None], (24, 1, 1))],
                                         axis=1)
        smpl_pose.append(joint_transform)
    smpl_pose = np.array(smpl_pose)

    intrinsics = np.tile(intrinsics, (len(smpl_pose), 1, 1))
    extrinsics = np.tile(extrinsics, (len(smpl_pose), 1, 1))
    smpl_pose = np.repeat(smpl_pose, n_camera, axis=0)
    frame_id = np.repeat(np.arange(0, n_frame, interval), n_camera, axis=0) + start_frame_idx

    cache = {"img": np_frames,
             "camera_intrinsic": intrinsics,
             "camera_rotation": extrinsics[:, :3, :3],
             "camera_translation": extrinsics[:, :3, 3:],
             "smpl_pose": smpl_pose,
             "frame_id": frame_id}

    os.makedirs(f"{DIR_PATH}/{person_id}/{prefix}{'_debug' * DEBUG}_cache_{n_frame}", exist_ok=True)
    with open(f"{DIR_PATH}/{person_id}/{prefix}{'_debug' * DEBUG}_cache_{n_frame}/cache.pickle", "wb") as f:
        pickle.dump(cache, f)


def preprocess(config):
    person_id = config["person_id"]
    n_camera = config["n_camera"]
    n_train_frame = config["n_train_frame"]
    n_test_frame = config["n_test_frame"]

    save_cache(person_id, n_camera, n_train_frame, prefix="train", start_frame_idx=0, interval=1)
    save_cache(person_id, n_camera, n_test_frame, prefix="test", start_frame_idx=n_train_frame, interval=10)

    global DEBUG
    DEBUG = True
    save_cache(person_id, n_camera, n_train_frame, prefix="train", start_frame_idx=0, interval=1)
    save_cache(person_id, n_camera, n_test_frame, prefix="test", start_frame_idx=n_train_frame, interval=10)


if __name__ == "__main__":
    DEBUG = False
    DIR_PATH = f"/data/unagi0/noguchi/dataset/NeuralActor"
    IMAGE_SIZE = 1024

    configs = [
        # {"person_id": "lan", "n_train_frame": 33605, "n_test_frame": 14235, "n_camera": 11},
        {"person_id": "marc", "n_train_frame": 38194, "n_test_frame": 23062, "n_camera": 12},
    ]
    for conf in configs:
        preprocess(conf)
