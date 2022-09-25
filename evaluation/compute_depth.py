import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
from dataset.dataset import SurrealPoseDepthDataset
from models.generator import TriNARFGenerator
from libraries.config import yaml_config

warnings.filterwarnings('ignore')

mse = nn.MSELoss()


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

        batchsize = len(minibatch["pose_3d"])

        z_dim = self.gen.config.z_dim * 3 if is_VAE else self.gen.config.z_dim * 4
        z = torch.cuda.FloatTensor(batchsize, z_dim).normal_()

        pose_to_camera = minibatch["pose_3d"].cuda(non_blocking=True)
        bone_length = minibatch["bone_length"].cuda(non_blocking=True)
        pose_to_world = minibatch["pose_3d_world"].cuda(non_blocking=True)
        intrinsic = minibatch["intrinsics"].cuda(non_blocking=True)
        gt_inv_depth = minibatch["img"]
        inv_intrinsic = torch.inverse(intrinsic)
        with torch.no_grad():
            _, _, gen_disparity = self.gen(pose_to_camera, pose_to_world, bone_length, z, inv_intrinsic,
                                           return_disparity=True, truncation_psi=args.truncation)
        self.i += 1
        return gen_disparity.reshape(-1, self.gen.size, self.gen.size), gt_inv_depth


def inv_depth_mse(gen_iterator):
    gen_all = []
    gt_all = []
    for gen, gt in tqdm(gen_iterator):
        gen_all.append(gen.cpu())
        gt_all.append(gt.cpu())
    gen_all = torch.cat(gen_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    return mse(gen_all, gt_all).item()


def main(config, batch_size=4, num_sample=10_000):
    size = config.dataset.image_size
    dataset_name = config.dataset.name
    train_dataset_config = config.dataset.train

    print("loading datasets")
    assert dataset_name == "human_v2"

    pose_prior_root = train_dataset_config.data_root
    print("pose prior:", pose_prior_root)
    pose_dataset = SurrealPoseDepthDataset(config_data)

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
        raise ValueError()
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
    disp_mse = inv_depth_mse(gen_iterator)

    out_dir = config.out_root
    out_name = config.out

    if args.truncation == 1:
        path = f"{out_dir}/result/{out_name}/disparity_mse.txt"
    else:
        path = f"{out_dir}/result/{out_name}/disparity_mse_trunc{args.truncation}.txt"
    with open(path, "w") as f:
        f.write(f"{disp_mse}")
    print(path, disp_mse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--surreal_path', type=str, default='data/surreal', help='path to surreal dataset')
    parser.add_argument('--config', type=str, default="configs/enarfgan_train/AIST/config.yml")
    parser.add_argument('--default_config', type=str, default="configs/enarfgan_train/default.yml")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--truncation', type=float, default=1)

    args = parser.parse_args()

    config_data = edict({"data_root": f"{args.surreal_path}/NARF_GAN_depth_cache"})

    config = yaml_config(args.config, args.default_config, num_workers=args.num_workers)
    is_VAE = "VAE" in args.config

    main(config, num_sample=10000)
