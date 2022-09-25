import os
import pickle
import random

import blosc
import numpy as np
from torch.utils.data import Dataset

from dataset.utils_3d import create_mask, pose_to_image_coord


class SMPLProperty:
    def __init__(self):
        self.is_blank = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1])

        self.num_bone = 19

        self.prev_seq = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 11, 9, 10,
                         11, 12, 13, 16, 17, 18, 20, 21, 22, 23, 24, 25]

        self.num_joint = self.num_bone  # same as num_bone for this class
        self.num_not_blank_bone = int(np.sum(self.is_blank == 0))  # number of bone which is not blank

        self.valid_keypoints = [i for i in range(len(self.is_blank)) if i not in self.prev_seq or self.is_blank[i] == 0]
        self.num_valid_keypoints = len(self.valid_keypoints)


class HumanDatasetBase(Dataset):
    def __init__(self, config, size=128, return_bone_params=True,
                 return_bone_mask=False, num_repeat_in_epoch=100, just_cache=False,
                 load_camera_intrinsics=False, return_mask=False):
        random.seed()
        self.size = size
        self.num_repeat_in_epoch = num_repeat_in_epoch

        self.return_bone_params = return_bone_params
        self.return_bone_mask = return_bone_mask
        self.return_mask = return_mask

        # read params from config
        self.data_root = config.data_root
        self.config = config
        self.load_camera_intrinsics = load_camera_intrinsics

        self.just_cache = just_cache
        self.parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                                 12, 13, 14, 16, 17, 18, 19, 20, 21])

        if self.return_bone_params:
            self.hpp = SMPLProperty()
            self.num_bone = 24
            self.num_bone_param = self.num_bone - 1
            self.num_valid_keypoints = self.hpp.num_valid_keypoints

    def __len__(self):
        return len(self.imgs) * self.num_repeat_in_epoch

    def get_bone_length(self, pose):
        coordinate = pose[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parents[1:]], axis=1)
        return length[:, None]

    def preprocess_img(self, img, bg=None):
        raise NotImplementedError()

    def random_sample(self):
        i = random.randint(0, len(self.imgs) - 1)
        return self.__getitem__(i)

    def get_intrinsic(self, i):
        raise NotImplementedError()

    def get_image(self, i):
        raise NotImplementedError()

    def add_blank_part(self, joint_mat_camera, joint_pos_image):
        idx = [0, 0] + list(range(10)) + [9, 9] + list(range(10, 24))
        return joint_mat_camera[:, idx], joint_pos_image[:, :, idx]

    def __getitem__(self, i):
        i = i % len(self.imgs)

        return_dict = {}

        img = self.get_image(i)
        if img.shape[0] == 4:  # last channel is mask
            mask = img[3]
            img = img[:3]
            return_dict["mask"] = mask
        elif self.return_mask:
            mask = (img != 255).any(axis=0).astype("int")
            return_dict["mask"] = mask

        img = self.preprocess_img(img)
        if not self.return_bone_params:
            if random.random() > 0.5:
                img = img[:, :, ::-1].copy()  # flip

        return_dict.update({"img": img, "idx": self.data_idx[i]})

        if self.return_bone_params:
            pose_to_camera = self.pose_to_camera[i].copy()
            pose_to_camera[:, 3, 3] = 1
            pose_to_world = self.pose_to_world[i].copy()
            pose_to_world[:, 3, 3] = 1
            bone_length = self.get_bone_length(pose_to_world)
            pose_translation = pose_to_camera[:, :3, 3:]  # (n_bone, 3, 1)

            intrinsics = self.get_intrinsic(i)
            pose_2d = np.matmul(intrinsics, pose_translation)  # (n_bone, 3, 1)
            pose_2d = pose_2d[:, :2, 0] / pose_2d[:, 2:, 0]  # (n_bone, 2)
            pose_2d = pose_2d.astype("float32")

            return_dict["pose_2d"] = pose_2d
            return_dict["pose_3d"] = pose_to_camera.astype("float32")
            return_dict["pose_3d_world"] = pose_to_world.astype("float32")
            return_dict["bone_length"] = bone_length.astype("float32")
            return_dict["intrinsics"] = intrinsics.astype("float32")  # (1, 3, 3)

            # just for compatibility
            return_dict["pose_to_camera"] = return_dict["pose_3d"]
            return_dict["pose_to_world"] = return_dict["pose_3d_world"]

            if self.return_bone_mask:
                joint_pos_image = pose_to_image_coord(pose_to_camera, intrinsics)

                # this is necessary for creating mask
                joint_mat_camera_, joint_pos_image_ = self.add_blank_part(pose_to_camera[None], joint_pos_image)

                _, bone_mask, _, _ = create_mask(self.hpp, joint_mat_camera_, joint_pos_image_,
                                                 self.size, thickness=0.5)

                return_dict["bone_mask"] = bone_mask.astype("float32")  # (1, 3, 3)
        return return_dict


class HumanDataset(HumanDatasetBase):
    def __init__(self, config, size=128, return_bone_params=True,
                 return_bone_mask=False, num_repeat_in_epoch=100, just_cache=False, load_camera_intrinsics=True,
                 return_mask=False, **kwargs):
        super(HumanDataset, self).__init__(config, size, return_bone_params, return_bone_mask, num_repeat_in_epoch,
                                           just_cache, load_camera_intrinsics, return_mask)

        self.focal_length = config.focal_length if hasattr(config, "focal_length") else None
        # common operations
        self.load_cache()
        if just_cache:
            return

        self.data_idx = np.arange(len(self.imgs))

    def load_cache(self):
        cache_path = f"{self.data_root}/cache.pickle"
        assert os.path.exists(cache_path)
        with open(cache_path, "rb") as f:
            data_dict = pickle.load(f)

        self.imgs = data_dict["img"]
        assert blosc.unpack_array(self.imgs[0]).shape[-1] == self.size
        if self.return_bone_params:
            camera_intrinsic = data_dict["camera_intrinsic"] if self.load_camera_intrinsics else None
            smpl_pose = data_dict["smpl_pose"]

            self.intrinsics = camera_intrinsic
            self.inv_intrinsics = np.linalg.inv(camera_intrinsic)
            self.pose_to_world = smpl_pose
            extrinsic = np.broadcast_to(np.eye(4), (len(self.imgs), 4, 4)).copy()

            if "camera_rotation" in data_dict:
                camera_rotation = data_dict["camera_rotation"]
                camera_translation = data_dict["camera_translation"]
                self.camera_rotation = camera_rotation
                extrinsic[:, :3, :3] = camera_rotation
                extrinsic[:, :3, 3:] = camera_translation
                self.pose_to_camera = np.matmul(extrinsic[:, None], self.pose_to_world)
            else:
                self.pose_to_camera = self.pose_to_world

            # load canonical pose
            canonical_pose_path = f"smpl_data/neutral_canonical.npy"
            if os.path.exists(canonical_pose_path):
                self.canonical_pose = np.load(canonical_pose_path)

            if "frame_id" in data_dict:
                self.frame_id = data_dict["frame_id"]

    def get_intrinsic(self, i):
        if self.focal_length is None:
            return self.intrinsics[i]
        else:
            intrinsic = np.array([[self.focal_length, 0, self.size / 2],
                                  [0, self.focal_length, self.size / 2],
                                  [0, 0, 1]], dtype="float32")
            return intrinsic

    def get_image(self, i):
        return blosc.unpack_array(self.imgs[i])

    def preprocess_img(self, img):
        img = (img / 127.5 - 1).astype("float32")  # 3 x 128 x 128
        return img


class SSODataset(HumanDataset):
    def __getitem__(self, i):
        return_dict = super(SSODataset, self).__getitem__(i)
        i = i % len(self.imgs)
        n_frames = self.config.n_frames
        return_dict["frame_id"] = self.frame_id[i]
        return_dict["frame_time"] = min(self.frame_id[i] / n_frames, 1)  # [0, 1]
        return_dict["camera_rotation"] = self.camera_rotation[i].astype("float32")
        return return_dict


class HumanPoseDataset(Dataset):
    def __init__(self, size=128, data_root="", just_cache=False, num_repeat_in_epoch=100):
        self.size = size
        self.data_root = data_root
        self.just_cache = just_cache
        self.num_repeat_in_epoch = num_repeat_in_epoch

        self.hpp = SMPLProperty()

        self.create_cache()

        self.num_bone = 24
        self.num_bone_param = self.num_bone - 1
        self.num_valid_keypoints = self.hpp.num_valid_keypoints

        self.parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                                 12, 13, 14, 16, 17, 18, 19, 20, 21])
        self.deterministic = False

    def __len__(self):
        return len(self.pose_to_world) * self.num_repeat_in_epoch

    def create_cache(self):
        cache_path = f"{self.data_root}/cache.pickle"
        assert os.path.exists(cache_path)
        with open(cache_path, "rb") as f:
            data_dict = pickle.load(f)

        camera_intrinsic = data_dict["camera_intrinsic"]
        smpl_pose = data_dict["smpl_pose"]

        self.intrinsics = camera_intrinsic
        self.inv_intrinsics = np.linalg.inv(camera_intrinsic)
        self.pose_to_world = smpl_pose
        extrinsic = np.broadcast_to(np.eye(4), (len(self.intrinsics), 4, 4)).copy()

        if "camera_rotation" in data_dict:
            camera_rotation = data_dict["camera_rotation"]
            camera_translation = data_dict["camera_translation"]
            extrinsic[:, :3, :3] = camera_rotation
            extrinsic[:, :3, 3:] = camera_translation
            self.pose_to_camera = np.matmul(extrinsic[:, None], self.pose_to_world)
        else:
            self.pose_to_camera = self.pose_to_world

        # load canonical pose
        canonical_pose_path = f"{self.data_root}/canonical.npy"
        if os.path.exists(canonical_pose_path):
            self.canonical_pose = np.load(canonical_pose_path)

    def add_blank_part(self, joint_mat_camera, joint_pos_image):
        idx = [0, 0] + list(range(10)) + [9, 9] + list(range(10, 24))
        return joint_mat_camera[:, idx], joint_pos_image[:, :, idx]

    def get_bone_length(self, pose):
        coordinate = pose[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parents[1:]], axis=1)
        return length[:, None]

    def scale_pose(self, pose, scale):
        pose[:, :3, 3] *= scale
        return pose

    def get_intrinsic(self, i):
        return self.intrinsics[i]

    def __getitem__(self, i):
        i = i % len(self.pose_to_world)
        joint_mat_world = self.pose_to_world[i]  # 24 x 4 x 4
        joint_mat_camera = self.pose_to_camera[i]

        bone_length = self.get_bone_length(joint_mat_world)

        intrinsics = self.get_intrinsic(i)

        joint_pos_image = pose_to_image_coord(joint_mat_camera, intrinsics)

        # this is necessary for creating mask
        joint_mat_camera_, joint_pos_image_ = self.add_blank_part(joint_mat_camera[None], joint_pos_image)

        disparity, mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, joint_mat_camera_, joint_pos_image_,
                                                                          self.size, thickness=0.5)
        return_dict = {
            "bone_mask": mask,  # size x size
            "pose_to_camera": joint_mat_camera.astype("float32"),  # num_joint x 4 x 4
            "bone_length": bone_length.astype("float32"),  # num_bone x 1
            "pose_to_world": joint_mat_world.astype("float32"),  # num_joint x 4 x 4
            "intrinsics": intrinsics.astype("float32"),  # (3, 3)
            "pose_2d": joint_pos_image[0].transpose()[:, :2]  # (num_bone, 2)
        }
        return return_dict


class SurrealPoseDepthDataset(HumanDataset):
    def __init__(self, config, size=128, return_bone_params=True,
                 return_bone_mask=False, num_repeat_in_epoch=1, just_cache=False, load_camera_intrinsics=True,
                 **kwargs):
        super(SurrealPoseDepthDataset, self).__init__(config, size, return_bone_params, return_bone_mask,
                                                      num_repeat_in_epoch,
                                                      just_cache, load_camera_intrinsics)

    def load_cache(self):
        cache_path = f"{self.data_root}/cache.pickle"
        assert os.path.exists(cache_path)
        with open(cache_path, "rb") as f:
            data_dict = pickle.load(f)

        self.imgs = data_dict["disparity"]
        assert blosc.unpack_array(self.imgs[0]).shape[-1] == self.size
        if self.return_bone_params:
            camera_intrinsic = data_dict["camera_intrinsic"] if self.load_camera_intrinsics else None
            smpl_pose = data_dict["smpl_pose"]

            self.intrinsics = camera_intrinsic
            self.inv_intrinsics = np.linalg.inv(camera_intrinsic)
            self.pose_to_world = smpl_pose

            self.pose_to_camera = self.pose_to_world

            # load canonical pose
            canonical_pose_path = f"smpl_data/neutral_canonical.npy"
            if os.path.exists(canonical_pose_path):
                self.canonical_pose = np.load(canonical_pose_path)

    def get_image(self, i):
        return blosc.unpack_array(self.imgs[i])

    def preprocess_img(self, img):
        img = img.astype("float32")  # 128 x 128
        return img
