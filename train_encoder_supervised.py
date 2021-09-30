import argparse
import os
import time

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from NARF.utils import record_setting, yaml_config, write
from dataset import THUmanDataset
from models.net import Encoder


def train(train_func, config):
    datasets, data_loaders = create_dataloader(config.dataset)
    train_func(config, datasets, data_loaders, rank=0, ddp=False)


def cache_dataset(config_dataset):
    create_dataset(config_dataset, just_cache=True)


def create_dataset(config_dataset, just_cache=False):
    size = config_dataset.image_size
    dataset_name = config_dataset.name

    train_dataset_config = config_dataset.train

    print("loading datasets")
    if dataset_name == "human":
        img_dataset = THUmanDataset(train_dataset_config, size=size, just_cache=just_cache)
    else:
        assert False
    return img_dataset


def create_dataloader(config_dataset):
    batch_size = config_dataset.bs
    shuffle = True
    drop_last = True
    num_workers = config_dataset.num_workers
    print("num_workers:", num_workers)

    img_dataset = create_dataset(config_dataset)
    loader_img = DataLoader(img_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                            drop_last=drop_last)

    return (img_dataset,), (loader_img,)


def prepare_models(img_dataset):
    enc = Encoder(img_dataset.parents, img_dataset.cp.intrinsics)
    return enc


def train_func(config, datasets, data_loaders, rank, ddp=False, world_size=1):
    # TODO(xiao): move outside
    torch.backends.cudnn.benchmark = True

    out_dir = config.out_root
    out_name = config.out
    if rank == 0:
        writer = tbx.SummaryWriter(f"{out_dir}/runs/{out_name}")
        os.makedirs(f"{out_dir}/result/{out_name}", exist_ok=True)
        record_setting(f"{out_dir}/result/{out_name}")

    num_iter = config.num_iter

    img_dataset, = datasets
    loader_img, = data_loaders

    enc = prepare_models(img_dataset)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    enc = enc.cuda(n_gpu)

    mse = nn.MSELoss()

    if ddp:
        enc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(enc)
        enc = nn.parallel.DistributedDataParallel(enc, device_ids=[n_gpu])

    enc_optimizer = optim.Adam(enc.parameters(), lr=5e-4, betas=(0, 0.99))

    iter = 0
    start_time = time.time()

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")

            if ddp:
                enc_module = enc.module
            else:
                enc_module = enc
            enc_module.load_state_dict(snapshot["enc"], strict=True)
            enc_optimizer.load_state_dict(snapshot["enc_opt"])
            iter = snapshot["iteration"]
            start_time = snapshot["start_time"]
            del snapshot

    while iter < num_iter:
        for i, minibatch in enumerate(loader_img):
            if (iter + 1) % 10 == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / iter} s/iter")
            enc.train()

            real_img = minibatch["img"].cuda(non_blocking=True).float()
            pose_2d = minibatch["pose_2d"].cuda(non_blocking=True).float()
            pose_3d = minibatch["pose_3d"].cuda(non_blocking=True).float()

            enc_optimizer.zero_grad()

            # reconstruction
            pose_3d_enc, _, _, _ = enc(real_img, pose_2d)  # (B, n_parts, 4, 4), (B, z_dim*4), (B, n_parts)
            scaled_pose_3d = enc.scale_pose(pose_3d[:, :, :3, 3:])

            loss = (mse(pose_3d[:, :, :3, :3], pose_3d_enc[:, :, :3, :3]) +  # rotation loss
                    mse(scaled_pose_3d, pose_3d_enc[:, :, :3, 3:]))  # translation loss

            if rank == 0:
                print(iter)
                write(iter, loss, "loss", writer)

            loss.backward()
            enc_optimizer.step()

            torch.cuda.empty_cache()

            if rank == 0:
                if iter == 10:
                    with open(f"{out_dir}/result/{out_name}/iter_10_succeeded.txt", "w") as f:
                        f.write("ok")
                if (iter + 1) % 200 == 0:
                    if ddp:
                        enc_module = enc.module
                    else:
                        enc_module = enc
                    save_params = {"iteration": iter,
                                   "start_time": start_time,
                                   "enc": enc_module.state_dict(),
                                   "enc_opt": enc_optimizer.state_dict()}
                    torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_latest.pth")
                    torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_{(iter // 50000 + 1) * 50000}.pth")

            torch.cuda.empty_cache()
            iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/NARF_GAN/THUman/20210903.yml")
    parser.add_argument('--default_config', type=str, default="configs/NARF_GAN_from_2d/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    train(train_func, config)
