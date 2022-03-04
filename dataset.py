import glob
import os
import pickle
import random

import blosc
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torch.utils.data import Dataset
from tqdm import tqdm

from NARF.models.utils_3d import THUmanPrior, CameraProjection, create_mask, pose_to_image_coord


class HumanDatasetBase(Dataset):
    def __init__(self, config, size=128, return_bone_params=True,
                 return_bone_mask=False, num_repeat_in_epoch=100, just_cache=False, load_camera_intrinsics=False):
        random.seed()
        self.size = size
        self.num_repeat_in_epoch = num_repeat_in_epoch

        self.return_bone_params = return_bone_params
        self.return_bone_mask = return_bone_mask

        # read params from config
        self.data_root = config.data_root
        self.config = config
        self.load_camera_intrinsics = load_camera_intrinsics

        self.just_cache = just_cache
        self.parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                                 12, 13, 14, 16, 17, 18, 19, 20, 21])

        if self.return_bone_params:
            self.cp = CameraProjection(size=size)
            self.hpp = THUmanPrior()
            self.num_bone = 24
            self.num_bone_param = self.num_bone - 1
            self.num_valid_keypoints = self.hpp.num_valid_keypoints
            self.intrinsics = self.cp.intrinsics

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

            if self.return_bone_mask:
                joint_pos_image = self.cp.pose_to_image_coord(pose_to_camera, intrinsics)

                # this is necessary for creating mask
                joint_mat_camera_, joint_pos_image_ = self.add_blank_part(pose_to_camera[None], joint_pos_image)

                _, bone_mask, _, _ = create_mask(self.hpp, joint_mat_camera_, joint_pos_image_,
                                                 self.size, thickness=0.5)

                return_dict["bone_mask"] = bone_mask.astype("float32")  # (1, 3, 3)
        return return_dict


class THUmanDataset(HumanDatasetBase):
    """THUman dataset"""

    def __init__(self, config, size=128, return_bone_params=True,
                 return_bone_mask=False, num_repeat_in_epoch=100, just_cache=False, load_camera_intrinsics=False):
        super(THUmanDataset, self).__init__(config, size, return_bone_params, return_bone_mask, num_repeat_in_epoch,
                                            just_cache, load_camera_intrinsics)

        # operations only for THUman
        self.n_mesh = config.n_mesh
        self.n_rendered_per_mesh = config.n_rendered_per_mesh
        self.n_imgs_per_mesh = config.n_imgs_per_mesh
        self.train = config.train

        self.imgs = self.cache_image()
        if self.return_bone_params:  # Need this here for just_cache==True
            self.pose_to_world_, self.pose_to_camera_, self.inv_intrinsics_ = self.cache_bone_params()
        if just_cache:
            return

        # [0, 1, ..., n_imgs_per_mesh-1, n_imgs_per_mesh, n_imgs_per_mesh+1, ...]
        data_idx = np.arange(self.n_mesh * self.n_imgs_per_mesh) % self.n_imgs_per_mesh + \
                   np.arange(self.n_mesh * self.n_imgs_per_mesh) // self.n_imgs_per_mesh * self.n_rendered_per_mesh

        if not self.train:
            data_idx = -1 - data_idx  # reverse order
        self.data_idx = data_idx

        self.imgs = self.imgs[data_idx]

        if self.return_bone_params:
            self.pose_to_world = self.pose_to_world_[data_idx]
            self.pose_to_camera = self.pose_to_camera_[data_idx]
            if self.inv_intrinsics_ is not None:
                self.inv_intrinsics = self.inv_intrinsics_[data_idx]

    def cache_image(self):
        if os.path.exists(f"{self.data_root}/render_{self.size}.npy"):
            if self.just_cache:
                return None

            imgs = np.load(f"{self.data_root}/render_{self.size}.npy")

        else:
            imgs = []
            img_paths = glob.glob(f"{self.data_root}/render_{self.size}/color/*.png")
            for path in tqdm(sorted(img_paths)):
                img = cv2.imread(path, -1)
                # img = cv2.resize(img, (self.size, self.size))
                imgs.append(img.transpose(2, 0, 1))

            imgs = np.array(imgs)

            np.save(f"{self.data_root}/render_{self.size}.npy", imgs)

        # load background images
        if self.config.background_path is not None:
            assert os.path.exists(self.config.background_path)
            self.background_imgs = np.load(self.config.background_path)

        return imgs

    def cache_bone_params(self):
        if os.path.exists(f"{self.data_root}/render_{self.size}_pose_to_world.npy"):
            if self.just_cache:
                return None, None, None

            pose_to_world = np.load(f"{self.data_root}/render_{self.size}_pose_to_world.npy")
            pose_to_camera = np.load(f"{self.data_root}/render_{self.size}_pose_to_camera.npy")

            if self.load_camera_intrinsics:
                inv_intrinsics = np.load(f"{self.data_root}/render_{self.size}_inv_intrinsics.npy")
            else:
                inv_intrinsics = None

        else:
            def save_pose(mode):
                pose = []
                pose_paths = glob.glob(f"{self.data_root}/render_{self.size}/color/pose_to_{mode}_*.npy")
                for path in tqdm(sorted(pose_paths)):
                    pose_ = np.load(path)
                    pose.append(pose_)

                pose = np.array(pose)

                np.save(f"{self.data_root}/render_{self.size}_pose_to_{mode}.npy", pose)
                return pose

            pose_to_world = save_pose("world")
            pose_to_camera = save_pose("camera")

            # camera intrinsics
            if self.load_camera_intrinsics:
                inv_intrinsics = []
                K_paths = glob.glob(f"{self.data_root}/render_{self.size}/color/camera_intrinsics_*.npy")
                for path in tqdm(sorted(K_paths)):
                    K_i = np.load(path)
                    inv_intrinsics.append(np.linalg.inv(K_i))
                inv_intrinsics = np.array(inv_intrinsics)

                np.save(f"{self.data_root}/render_{self.size}_inv_intrinsics.npy", inv_intrinsics)
            else:
                inv_intrinsics = None
        return pose_to_world, pose_to_camera, inv_intrinsics

    def get_intrinsic(self, i):
        return self.intrinsics

    def get_image(self, i):
        return self.imgs[i]

    def preprocess_img(self, img, bg=None):
        # bgra -> rgb, a
        mask = img[3:] / 255.
        img = img[:3]

        # blacken background
        img = img * mask
        if hasattr(self, "background_imgs"):
            if bg is None:
                bg_idx = random.randint(0, len(self.background_imgs) - 1)
                bg = self.background_imgs[bg_idx]
            img = img + bg[::-1] * (1 - mask)

        img = (img / 127.5 - 1).astype("float32")  # 3 x 128 x 128
        img = img[::-1].copy()  # BGR2RGB
        # mask = mask.astype("float32")  # 1 x 128 x 128
        return img


class HumanDataset(HumanDatasetBase):
    """Common human dataset class"""

    def __init__(self, config, size=128, return_bone_params=True,
                 return_bone_mask=False, num_repeat_in_epoch=100, just_cache=False, load_camera_intrinsics=True,
                 **kwargs):
        super(HumanDataset, self).__init__(config, size, return_bone_params, return_bone_mask, num_repeat_in_epoch,
                                           just_cache, load_camera_intrinsics)

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


class THUmanPoseDataset(Dataset):
    def __init__(self, size=128, data_root="", just_cache=False, num_repeat_in_epoch=100,
                 data_root2=None, data1_ratio=1, data2_ratio=0, crop_algo="narf"):
        """pose prior

        Args:
            size:
            data_root:
            just_cache:
            num_repeat_in_epoch:
            data_root2: if not None, mixed distribution of data1 and data2 is returned
            data1_ratio: mixing ratio
            data2_ratio: mixing ratio
            crop_algo: "narf" same algo as NARF with random scaling, "tight" tight cropping without scaling
        """
        self.size = size
        self.data_root = data_root
        self.data_root2 = data_root2
        self.data1_ratio = data1_ratio
        self.data2_ratio = data2_ratio
        self.crop_algo = crop_algo
        assert crop_algo in ["narf", "tight"]
        # TODO Extinguish narf algorithm

        self.just_cache = just_cache
        self.num_repeat_in_epoch = num_repeat_in_epoch

        assert data_root2 is not None or data2_ratio == 0
        assert data1_ratio + data2_ratio == 1.
        self.poses = self.create_cache(data_root)
        self.poses2 = self.create_cache(data_root2)
        assert self.poses2 is None or len(self.poses) >= len(
            self.poses2), "second pose prior should be larger than first prior"

        self.cp = CameraProjection(size=size)  # just holds camera intrinsics
        self.hpp = THUmanPrior()
        self.intrinsics = self.cp.intrinsics

        self.num_bone = 24
        self.num_bone_param = self.num_bone - 1
        self.num_valid_keypoints = self.hpp.num_valid_keypoints

        self.parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                                 12, 13, 14, 16, 17, 18, 19, 20, 21])
        self.deterministic = False

    def __len__(self):
        return len(self.poses) * self.num_repeat_in_epoch

    def create_cache(self, data_root):
        if data_root is None:
            return None
        if os.path.exists(f"{data_root}/bone_params_128.npy"):
            if self.just_cache:
                return None

            poses = np.load(f"{data_root}/bone_params_128.npy")

        else:
            poses = []
            pose_paths = glob.glob(f"{data_root}/render_128/bone_params/*.npy")
            for path in pose_paths:
                poses.append(np.load(path))

            poses = np.array(poses)

            np.save(f"{data_root}/bone_params_128.npy", poses)

        # load canonical pose
        if self.crop_algo == "tight":
            canonical_pose_path = f"smpl_data/neutral_canonical.npy"
        else:
            canonical_pose_path = f"smpl_data/male_canonical.npy"
        if os.path.exists(canonical_pose_path):
            self.canonical_pose = np.load(canonical_pose_path)
        return poses

    def sample_camera_mat(self, cam_t=None, theta=None, phi=None, angle=None):
        cam_distance = 5.0 if self.crop_algo == "tight" else 2.0
        if self.deterministic:
            if cam_t is None:
                cam_t = np.array((0, 0, cam_distance))

                theta = 0
                phi = 0
                angle = 0
            cam_r = np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
            cam_r = cam_r * angle
        else:
            if cam_t is None:
                cam_t = np.array((0, 0, cam_distance))

                theta = np.random.uniform(0, 0.3)
                phi = np.random.uniform(0, 2 * np.pi)
                angle = np.random.uniform(0, 2 * np.pi)
            cam_r = np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
            cam_r = cam_r * angle

        R = cv2.Rodrigues(cam_r)[0]
        T = np.vstack((np.hstack((R, cam_t.reshape(3, 1))), np.zeros((1, 4))))  # 4 x 4

        return T

    def add_blank_part(self, joint_mat_camera, joint_pos_image):
        idx = [0, 0] + list(range(10)) + [9, 9] + list(range(10, 24))
        return joint_mat_camera[:, idx], joint_pos_image[:, :, idx]

    def get_bone_length(self, pose):
        coordinate = pose[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parents[1:]], axis=1)
        return length[:, None]

    def preprocess(self, pose, scale=0.5):
        left_hip = 1
        right_hip = 2
        trans = -pose[[left_hip, right_hip], :3, 3].mean(axis=0)
        pose_copy = pose.copy()
        pose_copy[:, :3, 3] += trans[None,]
        pose_copy[:, :3, 3] *= scale
        return pose_copy

    def rotate_pose_in_place(self, pose, x_r, y_r, z_r):
        """rotates model (x-axis first, then y-axis, and then z-axis)"""
        mat_x, _ = cv2.Rodrigues(np.asarray([x_r, 0, 0], dtype=np.float32))
        mat_y, _ = cv2.Rodrigues(np.asarray([0, y_r, 0], dtype=np.float32))
        mat_z, _ = cv2.Rodrigues(np.asarray([0, 0, z_r], dtype=np.float32))
        mat = np.matmul(np.matmul(mat_x, mat_y), mat_z)
        T = np.eye(4)
        T[:3, :3] = mat

        pose = np.matmul(T, pose)

        return pose

    def rotate_pose_randomly(self, pose):  # random rotation
        y_rot = np.random.uniform(-np.pi, np.pi)
        x_rot = np.random.uniform(-0.3, 0.3)
        z_rot = np.random.uniform(-0.3, 0.3)
        # # mul(mat, mesh.v)
        pose = self.rotate_pose_in_place(pose, x_rot, y_rot, z_rot)
        return pose

    def transform_randomly(self, pose, scale=True):
        if self.deterministic:
            pose[:, :3, 3] *= 1.3
        else:
            pose = self.rotate_pose_randomly(pose)

            # random scale
            if scale:
                scale = np.random.uniform(1.0, 1.5)
            else:
                scale = 1.3
            pose[:, :3, 3] *= scale
        return pose

    def scale_pose(self, pose, scale):
        pose[:, :3, 3] *= scale
        return pose

    def get_intrinsic(self, i, pose_to_camera=None):
        if self.crop_algo == "narf":
            return self.intrinsics
        elif self.crop_algo == "tight":
            pose_to_camera = pose_to_camera.copy()
            pose_to_camera[15] = pose_to_camera[15] * 2 - pose_to_camera[12]
            joints_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            pose_to_camera = pose_to_camera[joints_to_use, :3, 3]

            projected_pose = pose_to_camera[:, :2] / pose_to_camera[:, 2:]  # projection to z=1
            top_left = projected_pose.min(axis=0)
            bottom_right = projected_pose.max(axis=0)
            center = (top_left + bottom_right) / 2
            size = (center - top_left).max() * 1.1
            x1, y1 = center - size
            x2, y2 = center + size

            f = self.size / (x2 - x1)
            cx = -self.size * x1 / (x2 - x1)
            cy = -self.size * y1 / (y2 - y1)
            intrinsics = np.array([[f, 0, cx],
                                   [0, f, cy],
                                   [0, 0, 1]], dtype="float32")
            return intrinsics
        else:
            raise ValueError()

    def __getitem__(self, i):
        if self.data1_ratio == 1.0:
            poses = self.poses
        else:
            # choose pose
            u = random.random()
            poses = self.poses if u < self.data1_ratio else self.poses2

        i = i % len(poses)
        joint_mat_world = poses[i]  # 24 x 4 x 4
        if self.crop_algo == "tight":
            joint_mat_world = self.preprocess(joint_mat_world, scale=1.0)
            joint_mat_world = self.rotate_pose_randomly(joint_mat_world)
        elif self.crop_algo == "narf":
            joint_mat_world = self.preprocess(joint_mat_world)
            joint_mat_world = self.transform_randomly(joint_mat_world)

        camera_mat = self.sample_camera_mat()

        bone_length = self.get_bone_length(joint_mat_world)

        joint_mat_camera = np.matmul(camera_mat, joint_mat_world)

        if self.crop_algo == "tight":
            intrinsics = self.get_intrinsic(i, joint_mat_camera)
        else:
            intrinsics = self.get_intrinsic(i)
        joint_pos_image = pose_to_image_coord(joint_mat_camera, intrinsics)
        joint_mat_camera = joint_mat_camera[None]

        # this is necessary for creating mask
        joint_mat_camera_, joint_pos_image_ = self.add_blank_part(joint_mat_camera, joint_pos_image)

        disparity, mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, joint_mat_camera_, joint_pos_image_,
                                                                          self.size, thickness=0.5)
        return_dict = {
            # "disparity": disparity,  # size x size
            "bone_mask": mask,  # size x size
            # "part_disparity":part_bone_disparity,  # num_joint x size x size
            "pose_to_camera": joint_mat_camera[0].astype("float32"),  # num_joint x 4 x 4
            # "keypoint": keypoint_mask,  # num_joint x size x size
            "bone_length": bone_length.astype("float32"),  # num_bone x 1
            "pose_to_world": joint_mat_world.astype("float32"),  # num_joint x 4 x 4
            "intrinsics": intrinsics.astype("float32"),  # (3, 3),
            "pose_2d": joint_pos_image[0].transpose()[:, :2]  # (num_bone, 2)
        }
        return return_dict

    def batch_same(self, num=100):
        idx = random.randint(0, len(self) - 1)
        d, m, p, j, k, l, w = self[idx]
        data = [d, m, p, j, w, k, l]
        batch = [data for i in range(num)]
        batch = zip(*batch)
        batch = (torch.tensor(np.stack(b)) for b in batch)
        return batch

    def batch_sequential_camera_pose(self, num=100, ood=False):  # ood = out of distribution
        joint_mat_world = self.poses[np.random.randint(0, len(self.poses))]  # 24 x 4 x 4
        joint_mat_world = self.preprocess(joint_mat_world)

        joint_mat_world_orig = self.transform_randomly(joint_mat_world)

        batch = []
        for i in range(num):
            if ood:
                camera_mat = self.sample_camera_mat(cam_t=np.array([0, 0, 1.5 + i / num]),  # 1.5 ~ 2.5
                                                    theta=np.pi * 2 * i / num,
                                                    phi=0,
                                                    angle=np.pi * 2 * i / num)
            else:
                camera_mat = self.sample_camera_mat(cam_t=np.array([0, 0, 2]), theta=0, phi=0,
                                                    angle=np.pi * 2 * i / num)

            bone_length = self.get_bone_length(joint_mat_world_orig)

            intrinsics = self.get_intrinsic(i)
            joint_mat_camera, joint_pos_image = self.cp.process_mat(joint_mat_world_orig, camera_mat, intrinsics)
            joint_mat_camera_, joint_pos_image_ = self.add_blank_part(joint_mat_camera, joint_pos_image)

            disparity, mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, joint_mat_camera_,
                                                                              joint_pos_image_, self.size)

            batch.append((disparity,  # size x size
                          mask,  # size x size
                          part_bone_disparity,  # num_joint x size x size
                          joint_mat_camera[0].astype("float32"),  # num_joint x 4 x 4
                          joint_mat_world[0].astype("float32"),  # num_joint x 4 x 4
                          keypoint_mask,  # num_joint x size x size
                          bone_length.astype("float32"),  # num_bone x 1
                          ))

        batch = zip(*batch)
        batch = (torch.tensor(np.stack(b)) for b in batch)
        return batch

    def batch_sequential_bone_param(self, num=100, num_pose=5, num_interpolate_param=None, loop=True,
                                    fix_pose=False, fix_bone_param=False):
        num_parts = 24
        if fix_pose:
            pose_idx = [np.random.randint(0, len(self.poses))] * num_pose
        else:
            pose_idx = np.random.permutation(len(self.poses))[:num_pose]

        joint_mat_world = self.poses[pose_idx]  # num_pose x 24 x 4 x 4
        joint_mat_world = np.stack([self.preprocess(mat) for mat in joint_mat_world])  # num_pose x 24 x 4 x 4

        if fix_bone_param:
            if not fix_pose:
                # joint_mat_world = np.stack([self.transform_randomly(mat, scale=False) for mat
                #                             in joint_mat_world])  # num_pose x 24 x 4 x 4
                joint_mat_world = np.stack([self.scale_pose(mat, 1.3) for i, mat
                                            in enumerate(joint_mat_world)])  # num_pose x 24 x 4 x 4
        else:
            if fix_pose:
                # # scale all
                # joint_mat_world = np.stack(
                #     [self.scale_pose(mat, 1.25 + np.sin(2 * np.pi * i / num_pose) * 0.25) for i, mat
                #      in enumerate(joint_mat_world)])  # num_pose x 24 x 4 x 4

                # paper visualization
                def scale_some_parts(pose, idx=12, factor=1.5):
                    t_pa = pose[self.parents[1:], :3, 3]
                    t_ch = pose[1:, :3, 3]
                    t_diff = t_ch - t_pa
                    if isinstance(factor, float):
                        t_diff[idx - 1] *= factor
                    else:
                        t_diff *= factor[:, None]

                    scaled_t = [pose[0, :3, 3]]
                    for i in range(1, pose.shape[0]):
                        scaled_t.append(scaled_t[self.parents[i]] + t_diff[i - 1])
                    scaled_t = np.stack(scaled_t, axis=0)  # num_bone x 3
                    pose[:, :3, 3] = scaled_t
                    return pose  # B x num_bone*3 x 1

                # # paper visualization
                # joint_mat_world = np.stack([self.scale_pose(mat, 1.2) for i, mat
                #                             in enumerate(joint_mat_world)])  # num_pose x 24 x 4 x 4
                # joint_mat_world = np.stack(
                #     [scale_some_parts(mat, factor=0.9 / 1.2 * i / (num_pose - 1) + 1.8 / 1.2 * (1 - i / (num_pose - 1)))
                #      for i, mat in enumerate(joint_mat_world)])  # num_pose x 24 x 4 x 4

                # git repo
                start_factor = np.random.uniform(1, 1.5, num_parts - 1)
                end_factor = np.random.uniform(1, 1.5, num_parts - 1)
                factor = np.linspace(start_factor, end_factor, num_pose)
                joint_mat_world = np.stack(
                    [scale_some_parts(mat.copy(), factor=factor_i)
                     for factor_i, mat in zip(factor, joint_mat_world)])  # num_pose x 24 x 4 x 4

            else:
                joint_mat_world = np.stack([self.transform_randomly(mat) for mat
                                            in joint_mat_world])  # num_pose x 24 x 4 x 4

        parent_mat = joint_mat_world[:, self.parents[1:]]  # num_pose x 23 x 4 x 4
        parent_mat = np.concatenate([np.tile(np.eye(4)[None, None], (num_pose, 1, 1, 1)), parent_mat], axis=1)

        child_translation = []
        for i in range(num_pose):
            trans_i = []
            for j in range(num_parts):
                trans_i.append(np.linalg.inv(parent_mat[i, j]).dot(joint_mat_world[i, j]))
            child_translation.append(np.array(trans_i))
        child_translation = np.array(child_translation)  # num_pose x 24 x 4 x 4

        # interpolation (slerp)
        interp_pose_to_world = []
        for i in range(num_parts):
            if loop:
                key_rots = np.concatenate([child_translation[:, i, :3, :3],
                                           child_translation[:1, i, :3, :3]], axis=0)  # repeat first
                key_times = np.arange(num_pose + 1)
                times = np.arange(num) * num_pose / num
                interp_trans = np.concatenate([
                    np.linspace(child_translation[j, i, :3, 3],
                                child_translation[(j + 1) % num_pose, i, :3, 3],
                                num // num_pose, endpoint=False) for j in range(num_pose)], axis=0)  # num x 3
            else:
                key_rots = child_translation[:, i, :3, :3]
                key_times = np.arange(num_pose)
                times = np.arange(num) * (num_pose - 1) / (num - 1)
                interp_trans = np.concatenate([
                    np.linspace(child_translation[j, i, :3, 3],
                                child_translation[(j + 1), i, :3, 3],
                                num // (num_pose - 1), endpoint=True) for j in range(num_pose - 1)], axis=0)  # num x 3
            slerp = Slerp(key_times, R.from_matrix(key_rots))
            interp_rots = slerp(times).as_matrix()  # num x 3 x 3

            interp_mat = np.concatenate([interp_rots, interp_trans[:, :, None]], axis=2)
            interp_mat = np.concatenate([interp_mat, np.tile(np.array([[[0, 0, 0, 1]]]), (num, 1, 1))],
                                        axis=1)  # num x 4 x 4
            interp_pose_to_world.append(interp_mat)
        interp_pose_to_world = np.array(interp_pose_to_world)  # num_parts x num x 4 x 4

        # fixed camera
        camera_mat = self.sample_camera_mat(cam_t=np.array([0, 0, 2]), theta=0, phi=0, angle=0)

        batch = []
        for i in range(num):
            mixed_pose_to_world = []
            for part_idx in range(num_parts):
                if self.parents[part_idx] == -1:
                    mat = np.eye(4)
                else:
                    mat = mixed_pose_to_world[self.parents[part_idx]]
                mat = mat.dot(interp_pose_to_world[part_idx, i])

                mixed_pose_to_world.append(mat)

            mixed_pose_to_world_orig = np.stack(mixed_pose_to_world)

            bone_length = self.get_bone_length(mixed_pose_to_world_orig)

            intrinsics = self.get_intrinsic(i)
            joint_mat_camera, joint_pos_image = self.cp.process_mat(mixed_pose_to_world_orig, camera_mat, intrinsics)
            joint_mat_camera_, joint_pos_image_ = self.add_blank_part(joint_mat_camera, joint_pos_image)

            disparity, mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, joint_mat_camera_,
                                                                              joint_pos_image_,
                                                                              self.size)

            batch.append((disparity,  # size x size
                          mask,  # size x size
                          part_bone_disparity,  # num_joint x size x size
                          joint_mat_camera[0].astype("float32"),  # num_joint x 4 x 4
                          mixed_pose_to_world_orig[0].astype("float32"),  # num_joint x 4 x 4
                          keypoint_mask,  # num_joint x size x size
                          bone_length.astype("float32"),  # num_bone x 1
                          ))

        batch = zip(*batch)
        batch = (torch.tensor(np.stack(b)) for b in batch)
        return batch


class HumanPoseDataset(THUmanPoseDataset):
    def __init__(self, size=128, data_root="", just_cache=False, num_repeat_in_epoch=100):
        self.size = size
        self.data_root = data_root
        self.just_cache = just_cache
        self.num_repeat_in_epoch = num_repeat_in_epoch

        self.cp = CameraProjection(size=size)  # just holds camera intrinsics
        self.hpp = THUmanPrior()

        # todo remove this
        self.intrinsics = self.cp.intrinsics

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

        joint_pos_image = self.cp.pose_to_image_coord(joint_mat_camera, intrinsics)

        # this is necessary for creating mask
        joint_mat_camera_, joint_pos_image_ = self.add_blank_part(joint_mat_camera[None], joint_pos_image)

        disparity, mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, joint_mat_camera_, joint_pos_image_,
                                                                          self.size, thickness=0.5)
        return_dict = {
            # "disparity": disparity,  # size x size
            "bone_mask": mask,  # size x size
            # "part_disparity":part_bone_disparity,  # num_joint x size x size
            "pose_to_camera": joint_mat_camera.astype("float32"),  # num_joint x 4 x 4
            # "keypoint": keypoint_mask,  # num_joint x size x size
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
