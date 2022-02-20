import os
import pickle
import sys

import blosc
import cv2
import numpy as np
import torch
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../../")
from utils.smpl_utils import get_pose


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


def save_cache(person_id, n_frame, views, image_paths, K, R, T, D, smpl, prefix="train", start_frame_idx=0, interval=1):
    cache_image = []
    cache_intrinsic = []
    cache_rotation = []
    cache_translation = []
    cache_smple_pose = []
    cache_frame_id = []

    for frame_id in tqdm(range(start_frame_idx, start_frame_idx + n_frame, interval)):
        smpl_idx = frame_id + 1 if person_id in ["313", "315"] else frame_id
        smpl_param = np.load(f"{DIR_PATH}/CoreView_{person_id}/new_params/{smpl_idx}.npy", allow_pickle=True)
        poses = smpl_param[()]['poses'].reshape(1, 24, 3)
        shapes = smpl_param[()]['shapes']

        trans = np.eye(4)
        trans[:3, :3] = cv2.Rodrigues(smpl_param[()]['Rh'])[0]
        trans[:3, 3] = smpl_param[()]['Th']

        with torch.no_grad():
            pose = get_pose(smpl, torch.tensor(shapes), body_pose=torch.tensor(poses[:, 1:]).float(),
                            global_orient=torch.tensor(poses[:, 0:1, :]).float()).numpy()[0]
            pose_to_world = np.matmul(trans, pose)
            # pose_to_camera = np.matmul(cam_trans[:, None], pose_to_world)
            # pose_image = np.matmul(np.array(K)[:, None], pose_to_camera[:, :, :3, 3:])
            # pose_image = pose_image[:, :, :2, 0] / pose_image[:, :, 2:, 0]

        for view in views:
            img_path = image_paths[frame_id]['ims'][view]
            image = cv2.imread(f"{DIR_PATH}/CoreView_{person_id}/{img_path}")
            image = cv2.undistort(image, K[view], D[view])

            mask = cv2.imread(f"{DIR_PATH}/CoreView_{person_id}/mask/{img_path[:-3]}png")
            mask = cv2.undistort(mask, K[view], D[view])

            # resize
            h, w, _ = image.shape
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)[:, :, :1]
            K_new = K[view].copy()
            K_new[:2] *= IMAGE_SIZE / h
            # mask background
            image = image * (mask > 0)

            # concat
            image = np.concatenate([image[:, :, ::-1], (mask > 0).astype("uint8")], axis=-1)

            cache_image.append(blosc.pack_array(image.transpose(2, 0, 1)))
            cache_intrinsic.append(K_new)
            cache_rotation.append(R[view])
            cache_translation.append(T[view])
            cache_smple_pose.append(pose_to_world)
            cache_frame_id.append(frame_id)

    cache = {"img": np.array(cache_image, dtype="object"),
             "camera_intrinsic": np.array(cache_intrinsic),
             "camera_rotation": np.array(cache_rotation),
             "camera_translation": np.array(cache_translation),
             "smpl_pose": np.array(cache_smple_pose),
             "frame_id": np.array(cache_frame_id),
             }
    os.makedirs(f"{DIR_PATH}/CoreView_{person_id}/{prefix}_cache_{n_frame}", exist_ok=True)
    with open(f"{DIR_PATH}/CoreView_{person_id}/{prefix}_cache_{n_frame}/cache.pickle", "wb") as f:
        pickle.dump(cache, f)


def preprocess(config):
    smpl = SMPL(model_path=SMPL_MODEL_PATH, gender='NEUTRAL', batch_size=1)

    person_id = config["person_id"]
    K, R, T, D, cam_trans, image_paths, n_camera = read_annots(person_id)

    n_train_frame = config["n_train_frame"]
    n_test_frame = config["n_test_frame"]
    training_view = config["training_view"]
    testing_view = [i for i in range(n_camera) if i not in training_view]

    save_cache(person_id, n_train_frame, training_view, image_paths, K, R, T, D, smpl, prefix="train")
    save_cache(person_id, n_train_frame, testing_view, image_paths, K, R, T, D, smpl, prefix="test_novel_view",
               interval=30)
    save_cache(person_id, n_test_frame, testing_view, image_paths, K, R, T, D, smpl, prefix="test_novel_pose",
               interval=30)


if __name__ == "__main__":
    SMPL_MODEL_PATH = "../../smpl_data"
    DIR_PATH = f"/data/unagi0/noguchi/dataset/animatable_nerf_zju"
    IMAGE_SIZE = 512

    configs = [
        {"person_id": "313", "n_train_frame": 60, "n_test_frame": 1000, "training_view": [0, 6, 12, 18]},
        {"person_id": "315", "n_train_frame": 2085, "n_test_frame": 100, "training_view": [0, 6, 12, 18]},
        {"person_id": "377", "n_train_frame": 517, "n_test_frame": 100, "training_view": [0, 6, 12, 18]},
    ]
    for conf in configs:
        preprocess(conf)
