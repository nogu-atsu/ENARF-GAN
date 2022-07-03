# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import warnings

import numpy as np
import torch
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results)
from mmpose.datasets import DatasetInfo
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

sys.path.append(".")
from dependencies.config import yaml_config
from dataset.dataset import THUmanPoseDataset, HumanPoseDataset
from models.generator import NARFNRGenerator, TriNARFGenerator

warnings.filterwarnings('ignore')


class DetectPose:
    def __init__(self):
        self.det_config = "/home/mil/noguchi/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
        self.det_checkpoint = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        self.pose_config = "/home/mil/noguchi/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hrnet_w32_mpii_256x256_dark.py"
        self.pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark-f1601c5b_20200927.pth"
        self.device = 'cuda'
        self.det_cat_id = 1
        self.bbox_thr = 0.3
        self.kpt_thr = 0.3

        self.det_model = init_detector(
            self.det_config, self.det_checkpoint, device=self.device.lower())
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(
            self.pose_config, self.pose_checkpoint, device=self.device.lower())

        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)

    def __call__(self, img):

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, img[:, :, ::-1])

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, self.det_cat_id)

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # print(pose_results)
        return pose_results


class GenIterator:
    def __init__(self, gen, dataloader, num_sample):
        self.gen = gen
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.num_sample = num_sample
        self.i = 0
        assert len(dataloader.dataset) > 0
        assert len(dataloader.dataset) >= num_sample

        self.data = self.generator()

        self.gen.train()  # avoid fixed cropping of background

    def __iter__(self):
        return self

    def __len__(self):
        return (self.num_sample - 1) // self.batch_size + 1

    def generator(self):
        for minibatch in self.dataloader:
            yield minibatch

    def __next__(self):
        if self.i == len(self):
            raise StopIteration()
        minibatch = self.data.__next__()  # randomly sample latent

        batchsize = len(minibatch["pose_to_camera"])

        z_dim = self.gen.config.z_dim * 3 if is_VAE else self.gen.config.z_dim * 4
        z = torch.cuda.FloatTensor(batchsize, z_dim).normal_()

        pose_to_camera = minibatch["pose_to_camera"].cuda(non_blocking=True)
        bone_length = minibatch["bone_length"].cuda(non_blocking=True)
        pose_to_world = minibatch["pose_to_world"].cuda(non_blocking=True)
        intrinsic = minibatch["intrinsics"].cuda(non_blocking=True)
        pose_2d = minibatch["pose_2d"]
        inv_intrinsic = torch.inverse(intrinsic)
        with torch.no_grad():
            fake_img, _, _, _ = self.gen(pose_to_camera, pose_to_world, bone_length, z, inv_intrinsic,
                                         truncation_psi=args.truncation)
        self.i += 1
        return torch.clamp(fake_img, -1, 1).cpu().numpy(), pose_2d.numpy()


def detect_pose_from_generated(gen_iterator):
    detect_pose = DetectPose()

    gt_pose = []
    pred_pose = []
    for img, pose_2d in tqdm(gen_iterator):
        with torch.no_grad():
            # color range is [0, 255]
            img = (img * 127.5 + 127.5).astype("uint8")
            for p2, im in zip(pose_2d, img):
                pose_results = detect_pose(im.transpose(1, 2, 0))

                gt_pose.append(p2)
                pred_pose.append(pose_results)
    return gt_pose, pred_pose, img.shape[0]


def calc_error(gt_pose, pred_pose, size, kpts_thres=0.3):
    joint_error = []
    for gt, pred in zip(gt_pose, pred_pose):
        invalid = ((gt < 0) | (gt >= size)).any(axis=1)
        gt_ = gt.copy()
        gt_[invalid] = 1e5
        gt_selected = gt[smpl_idx]
        if len(pred) > 0:  # person is detected
            errors = []
            for pred_kpts in pred:
                kpts = pred_kpts['keypoints']
                invalid = kpts[:, 2] < kpts_thres
                kpts = kpts[:, :2].copy()
                kpts[invalid] = 1e10  # mask keypoints not detected
                pred_selected = kpts[mpii_idx]
                error = (np.linalg.norm(gt_selected - pred_selected, axis=1) /
                         np.linalg.norm(gt[head] - gt[neck]))
                errors.append(error)
            error = np.array(errors).min(axis=0)  # (n_kpts, )
        else:
            error = np.ones(len(smpl_idx)) * 1e10
        joint_error.append(error)

    joint_error = np.array(joint_error)
    return joint_error


def pck(joint_error):
    # import pdb
    # pdb.set_trace()
    pckh = np.mean(joint_error < 0.5, axis=0)

    out_dir = config.out_root
    out_name = config.out

    if args.truncation != 1:
        suffix = f"trunc{args.truncation}"
    else:
        suffix = ""

    path = f"{out_dir}/result/{out_name}/pckh_{suffix}.npy"
    np.save(path, pckh)
    print(path, pckh)


def main(config, batch_size=4, num_sample=10_000):
    size = config.dataset.image_size
    dataset_name = config.dataset.name
    train_dataset_config = config.dataset.train
    just_cache = False

    print("loading datasets")
    if dataset_name == "human":
        pose_prior_root = train_dataset_config.data_root

        print("pose prior:", pose_prior_root)
        pose_dataset = THUmanPoseDataset(size=size, data_root=pose_prior_root,
                                         just_cache=just_cache)
    elif dataset_name == "human_v2":
        pose_prior_root = train_dataset_config.data_root
        print("pose prior:", pose_prior_root)
        pose_dataset = HumanPoseDataset(size=size, data_root=pose_prior_root,
                                        just_cache=just_cache)

    else:
        assert False
    # pose_dataset.num_repeat_in_epoch = 1
    loader_pose = DataLoader(pose_dataset, batch_size=batch_size, num_workers=2, shuffle=True,
                             drop_last=True)

    gen_config = config.generator_params

    if gen_config.use_triplane:
        gen = TriNARFGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                               num_bone_param=pose_dataset.num_bone_param,
                               parent_id=pose_dataset.parents,
                               black_background=is_VAE)
        gen.register_canonical_pose(pose_dataset.canonical_pose)
        gen.to("cuda")
    else:
        gen = NARFNRGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                              num_bone_param=pose_dataset.num_bone_param, parent_id=pose_dataset.parents).to("cuda")
    out_dir = config.out_root
    out_name = config.out
    iteration = args.iteration if args.iteration > 0 else "latest"
    path = f"{out_dir}/result/{out_name}/snapshot_{iteration}.pth"
    if os.path.exists(path):
        snapshot = torch.load(path, map_location="cuda")
        for k in list(snapshot["gen"].keys()):
            if "activate.bias" in k:
                snapshot["gen"][k[:-13] + "bias"] = snapshot["gen"][k].reshape(1, -1, 1, 1)
                del snapshot["gen"][k]
        gen.load_state_dict(snapshot["gen"], strict=False)
    else:
        assert False, "pretrained model is not loading"

    gen_iterator = GenIterator(gen, loader_pose, num_sample=num_sample)
    gt_pose, pred_pose, img_size = detect_pose_from_generated(gen_iterator)
    joint_error = calc_error(gt_pose, pred_pose, img_size, kpts_thres=0.3)
    pck(joint_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/NARF_GAN/THUman/20210903.yml")
    parser.add_argument('--default_config', type=str, default="configs/NARF_GAN/default.yml")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--truncation', type=float, default=1)

    args = parser.parse_args()

    smpl_idx = [8, 5, 2, 1, 4, 7, 12, 21, 19, 17, 16, 18, 20]
    mpii_idx = [0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14, 15]
    neck = 8
    head = 9
    name = ["rank", "rkne", "rhip", "lhip", "lkne", "lank", "neck", "rwri", "relb", "rsho", "lsho", "lelb", "lwri"]

    config = yaml_config(args.config, args.default_config, num_workers=args.num_workers)
    is_VAE = "VAE" in args.config

    main(config, num_sample=10000)
