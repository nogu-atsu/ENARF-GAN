import argparse
import os
import time

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from NARF.utils import record_setting, yaml_config, write
from dataset import THUmanDataset
from models.loss import nerf_patch_loss
from models.net import NeRFNRGenerator, Encoder
from utils.mask_utils import create_bone_mask


def train(train_func, config):
    datasets, data_loaders = create_dataloader(config.dataset)
    train_func(config, datasets, data_loaders, rank=0, ddp=False)


def cache_dataset(config_dataset):
    create_dataset(config_dataset, just_cache=True)


def create_dataset(config_dataset, just_cache=False):
    size = config_dataset.image_size
    dataset_name = config_dataset.name

    train_dataset_config = config_dataset.train
    test_dataset_config = config_dataset.test

    print("loading datasets")
    if dataset_name == "human":
        img_dataset = THUmanDataset(train_dataset_config, size=size, just_cache=just_cache, multiview=True)
        test_img_dataset = THUmanDataset(test_dataset_config, size=size, just_cache=just_cache,
                                         num_repeat_in_epoch=1, multiview=True)
    else:
        assert False
    return img_dataset, test_img_dataset


def create_dataloader(config_dataset):
    batch_size = config_dataset.bs
    shuffle = True
    drop_last = True
    num_workers = config_dataset.num_workers
    print("num_workers:", num_workers)

    img_dataset, test_img_dataset = create_dataset(config_dataset)
    loader_img = DataLoader(img_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                            drop_last=drop_last)
    test_loader_img = DataLoader(test_img_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                 drop_last=False)

    return (img_dataset, test_img_dataset), (loader_img, test_loader_img)


def prepare_models(enc_config, gen_config, img_dataset, size):
    enc = Encoder(enc_config, img_dataset.parents, img_dataset.cp.intrinsics)
    gen = NeRFNRGenerator(gen_config, size, img_dataset.cp.intrinsics, num_bone=img_dataset.num_bone,
                          num_bone_param=img_dataset.num_bone_param)
    return enc, gen


def evaluate(enc, gen, test_loader):
    print("validation")
    mse = nn.MSELoss()
    enc.eval()
    gen.eval()
    loss_rotation = 0
    loss_translation = 0
    loss_recon = 0
    with torch.no_grad():
        for minibatch in tqdm(test_loader):
            real_img = minibatch["img"].cuda(non_blocking=True).float()
            pose_2d = minibatch["pose_2d"].cuda(non_blocking=True).float()
            pose_3d_gt = minibatch["pose_3d"].cuda(non_blocking=True).float()
            img_other_view = minibatch["img_other_view"].cuda(non_blocking=True).float()
            relative_rotation = minibatch["relative_rotation"].cuda(non_blocking=True).float()
            bs = real_img.shape[0]

            pose_3d, z, bone_length, intrinsic = enc(real_img, pose_2d)
            scaled_pose_3d_gt = enc.scale_pose(pose_3d_gt[:, :, :3, 3:])
            loss_rotation += mse(pose_3d[:, :, :3, :3], pose_3d_gt[:, :, :3, :3]) * bs
            loss_translation += mse(pose_3d[:, :, :3, 3:], scaled_pose_3d_gt) * bs

            rotated_pose_3d = torch.matmul(relative_rotation[:, None], pose_3d)
            fake_img, fake_low_res_mask = gen(rotated_pose_3d, None, bone_length, z)

            loss_recon += mse(img_other_view, fake_img)

    loss_rotation = loss_rotation / len(test_loader.dataset)
    loss_translation = loss_translation / len(test_loader.dataset)
    loss_recon = loss_recon / len(test_loader.dataset)

    loss_dict = {}
    loss_dict["loss_rotation_val"] = loss_rotation
    loss_dict["loss_translation_val"] = loss_translation
    loss_dict["loss_recon"] = loss_recon

    enc.train()
    gen.train()

    return loss_dict


def train_func(config, datasets, data_loaders, rank, ddp=False, world_size=1):
    # TODO(xiao): move outside
    torch.backends.cudnn.benchmark = True

    out_dir = config.out_root
    out_name = config.out
    if rank == 0:
        writer = tbx.SummaryWriter(f"{out_dir}/runs/{out_name}")
        os.makedirs(f"{out_dir}/result/{out_name}", exist_ok=True)
        record_setting(f"{out_dir}/result/{out_name}")

    size = config.dataset.image_size
    num_iter = config.num_iter

    img_dataset, test_img_dataset = datasets
    loader_img, test_loader_img = data_loaders

    enc, gen = prepare_models(config.encoder_params, config.generator_params, img_dataset, size)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    enc = enc.cuda(n_gpu)
    gen = gen.cuda(n_gpu)

    mse = nn.MSELoss()

    if ddp:
        enc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(enc)
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        enc = nn.parallel.DistributedDataParallel(enc, device_ids=[n_gpu])
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[n_gpu])

    bone_loss_func = nerf_patch_loss

    enc_optimizer = optim.Adam(enc.parameters(), lr=5e-4, betas=(0, 0.99))
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3, betas=(0, 0.99))

    iter = 0
    start_time = time.time()

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")

            if ddp:
                enc_module = enc.module
                gen_module = gen.module
            else:
                enc_module = enc
                gen_module = gen
            enc_module.load_state_dict(snapshot["enc"], strict=True)
            gen_module.load_state_dict(snapshot["gen"], strict=True)
            enc_optimizer.load_state_dict(snapshot["enc_opt"])
            gen_optimizer.load_state_dict(snapshot["gen_opt"])
            iter = snapshot["iteration"]
            start_time = snapshot["start_time"]
            del snapshot
    while iter < num_iter:
        for i, minibatch in enumerate(loader_img):
            if (iter + 1) % 10 == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / iter} s/iter")
            enc.train()
            gen.train()

            real_img = minibatch["img"].cuda(non_blocking=True).float()
            pose_2d = minibatch["pose_2d"].cuda(non_blocking=True).float()
            img_other_view = minibatch["img_other_view"].cuda(non_blocking=True).float()
            relative_rotation = minibatch["relative_rotation"].cuda(non_blocking=True).float()

            enc_optimizer.zero_grad()
            gen_optimizer.zero_grad()

            # reconstruction
            pose_3d, z, bone_length, intrinsic = enc(real_img, pose_2d)

            rotated_pose_3d = torch.matmul(relative_rotation[:, None], pose_3d)
            fake_img, fake_low_res_mask = gen(rotated_pose_3d, None, bone_length, z)

            loss_dict = {}
            loss_recon = mse(img_other_view, fake_img)
            loss_dict["loss_recon"] = loss_recon

            bone_mask = create_bone_mask(img_dataset.parents, pose_3d, size, intrinsic)
            loss_bone = bone_loss_func(fake_low_res_mask, bone_mask)
            loss_dict["loss_bone"] = loss_bone

            loss = loss_recon + loss_bone

            if rank == 0:
                if iter % 100 == 0:
                    print(iter)
                    for k, v in loss_dict.items():
                        write(iter, v, k, writer, True)

            loss.backward()
            enc_optimizer.step()

            torch.cuda.empty_cache()

            if rank == 0:
                if iter == 10:
                    with open(f"{out_dir}/result/{out_name}/iter_10_succeeded.txt", "w") as f:
                        f.write("ok")
                if (iter + 1) % 1000 == 0:
                    loss_dict_val = evaluate(enc, gen, test_loader_img)
                    for k, v in loss_dict_val.items():
                        write(iter, v, k, writer, True)

                if (iter + 1) % 200 == 0:
                    if ddp:
                        enc_module = enc.module
                        gen_module = gen.module
                    else:
                        enc_module = enc
                        gen_module = gen
                    save_params = {"iteration": iter,
                                   "start_time": start_time,
                                   "enc": enc_module.state_dict(),
                                   "gen": gen_module.state_dict(),
                                   "enc_opt": enc_optimizer.state_dict(),
                                   "gen_opt": gen_optimizer.state_dict()}
                    torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_latest.pth")
                    torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_{(iter // 5000 + 1) * 5000}.pth")

            torch.cuda.empty_cache()
            iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/Encoder/THUman/20210903.yml")
    parser.add_argument('--default_config', type=str, default="configs/Encoder_Generator/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    train(train_func, config)
