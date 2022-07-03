import argparse
import os
import sys

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import (init_pose_model, process_mmdet_results,
                         inference_top_down_pose_model, vis_pose_result)
from mmpose.core.evaluation.top_down_eval import (keypoint_pck_accuracy)
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

sys.path.append(".")
from evaluation.compute_PCK_fixed import GenIterator
from dependencies.config import yaml_config
from dataset.dataset import HumanDataset
from models.generator import NARFNRGenerator, TriNARFGenerator


# ----------------------------------------------------------------------------

def compute_pck(opts, max_real, num_gen):
    pose_model, det_model = load_mmcv_models()

    pck = compute_pck_for_dataset(opts=opts, max_items=num_gen, pose_model=pose_model, det_model=det_model)
    if opts.rank != 0:
        return float('nan')
    return float(pck)


def load_mmcv_models():
    pose_config = "/home/mil/noguchi/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hrnet_w32_mpii_256x256_dark.py"
    pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark-f1601c5b_20200927.pth"
    det_config = "/home/mil/noguchi/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
    det_checkpoint = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    # initialize pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint)
    # initialize detector
    det_model = init_detector(det_config, det_checkpoint)

    return pose_model, det_model


def compute_kpts(images, pose_model, det_model, mode):
    kps = []
    failed = 0
    for i in range(images.shape[0]):
        img = images[i].transpose(1, 2, 0)
        img = np.clip((img[:, :, ::-1] * 127.5 + 127.5), 0, 255).astype("uint8")

        mmdet_results = inference_detector(det_model, img)
        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=pose_model.cfg.data.test.type)

        # debug test to see how accurate kpt estimation is
        if False:
            vis_result = vis_pose_result(
                pose_model,
                img,
                pose_results,
                dataset=pose_model.cfg.data.test.type,
                show=False)
            vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
            cv2.imwrite(f'test_{mode}.png', vis_result)

        # no human even detected:
        if len(pose_results) == 0:
            kps.append(-1e8 * np.ones((16, 3)))
            failed = 1
        else:
            kps.append(pose_results[0]['keypoints'])

    return np.stack(kps, 0), failed


def compute_pck_for_dataset(gen_iter, batch_size=64, batch_gen=None, pose_model=None, det_model=None, max_items=0):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0
    batch_gen = 8

    # SOME NETWORK INITIALIZATION

    # Initialize.
    teval_imgs = 0
    # Main loop.
    hits = 0
    total = 0

    with tqdm(total=max_items) as pbar:
        while teval_imgs < max_items:
            for _i in range(batch_size // batch_gen):
                # GET GT AND GENERATED RESULT IN THE SAME CAMERA POSE AND SMPL POSE
                pred_img, _, gt_img = gen_iter.__next__()
                gt_kpts, gt_failed = compute_kpts(gt_img, pose_model, det_model, 'gt')
                pred_kpts, pred_failed = compute_kpts(pred_img, pose_model, det_model, 'pred')
                gt_scores = gt_kpts[..., -1]
                gt_kpts = gt_kpts[..., :2]
                pred_scores = pred_kpts[..., -1]
                pred_kpts = pred_kpts[..., :2]

                mask = np.logical_and((gt_scores > 0.8), (pred_scores > 0.8))
                # mask = (gt_scores > 0.8)
                thr = 0.5
                interocular = np.linalg.norm(gt_kpts[:, 0, :] - gt_kpts[:, 1, :], axis=1, keepdims=True)
                normalize = np.tile(interocular, [1, 2])

                oe = keypoint_pck_accuracy(pred_kpts, gt_kpts, mask, thr, normalize)
                hits += oe[1] * oe[2] * pred_kpts.shape[0]
                total += oe[2] * pred_kpts.shape[0]

                teval_imgs += batch_gen
                pbar.update(batch_gen)

    print(f'Total: {total}')
    return float(hits) / float(total)


def main(config, batch_size=4, num_sample=10_000):
    size = config.dataset.image_size
    dataset_name = config.dataset.name
    train_dataset_config = config.dataset.train
    just_cache = False

    print("loading datasets")
    if dataset_name == "human":
        raise ValueError("human dataset is not supported")
    elif dataset_name == "human_v2":
        pose_dataset = HumanDataset(train_dataset_config, size=size, num_repeat_in_epoch=1,
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

    gen_iterator = GenIterator(gen, loader_pose, num_sample, is_VAE, args.truncation)

    pose_model, det_model = load_mmcv_models()
    pck = compute_pck_for_dataset(gen_iterator, batch_gen=batch_size, pose_model=pose_model, det_model=det_model,
                                  max_items=num_sample)
    print(pck)
    # import pickle
    # with open(f"tmp_pose_array.pkl", "wb") as f:
    #     pickle.dump({"gt_pose": gt_pose,
    #                  "pred_pose": pred_pose}, f)
    # joint_error = calc_error(gt_pose, pred_pose, img_size, kpts_thres=0.3)
    # pck(joint_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/NARF_GAN/THUman/20210903.yml")
    parser.add_argument('--default_config', type=str, default="configs/NARF_GAN/default.yml")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--truncation', type=float, default=1)

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, num_workers=args.num_workers)
    is_VAE = "VAE" in args.config

    main(config, num_sample=10000)
