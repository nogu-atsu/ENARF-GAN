import argparse
import os
import time
import warnings

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

from utils.train_utils import record_setting, write
from utils.config import yaml_config
from utils.visualization_utils import save_img
from dataset import THUmanDataset, HumanDataset
from models.loss import nerf_patch_loss
from models.net import NeRFNRGenerator, TriNeRFGenerator

warnings.filterwarnings('ignore')

mse = nn.MSELoss(reduction="sum")


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
        dataset = THUmanDataset(train_dataset_config, size=size, return_bone_params=True,
                                just_cache=just_cache)

    elif dataset_name == "human_v2":
        # TODO mixed prior
        dataset = HumanDataset(train_dataset_config, size=size, return_bone_params=True,
                               return_bone_mask=True, just_cache=just_cache)

    elif dataset_name == "human_tight":
        dataset = HumanDataset(train_dataset_config, size=size, return_bone_params=True,
                               just_cache=just_cache)

    else:
        assert False
    return dataset


def create_dataloader(config_dataset):
    batch_size = config_dataset.bs
    shuffle = True
    drop_last = True
    num_workers = config_dataset.num_workers
    print("num_workers:", num_workers)

    dataset = create_dataset(config_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                        drop_last=drop_last)

    return dataset, loader


def prepare_models(gen_config, pose_dataset, size):
    if gen_config.use_triplane:
        gen = TriNeRFGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                               num_bone_param=pose_dataset.num_bone_param,
                               parent_id=pose_dataset.parents,
                               black_background=True)
        gen.register_canonical_pose(pose_dataset.canonical_pose)
    else:
        gen = NeRFNRGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                              num_bone_param=pose_dataset.num_bone_param, parent_id=pose_dataset.parents)
    enc = models.resnet50(pretrained=True)
    enc.fc = nn.Linear(2048, gen_config.z_dim * 6)
    return gen, enc


def train_step(iter, gen, pose_to_camera, pose_to_world, bone_length, inv_intrinsic,
               enc, gen_optimizer, enc_optimizer,
               rank, writer, real_img, real_mask, n_accum_step=1):
    # randomly sample latent
    enc.eval()

    batchsize = len(real_img) // n_accum_step

    loss_recon = 0
    loss_kl = 0
    fake_img = []
    for i in range(0, len(real_img), batchsize):
        z = enc(real_img[i:i + batchsize])
        z_dim = z.shape[1] // 2
        z_mean, z_std = z[:, :z_dim], F.softplus(z[:, z_dim:] / 2)
        epsilon = torch.randn(z_mean.shape, device=z_mean.device)
        z = z_mean + epsilon * z_std

        pose_to_camera_i = pose_to_camera[i:i + batchsize]
        pose_to_world_i = pose_to_world[i:i + batchsize]
        bone_length_i = bone_length[i:i + batchsize]
        inv_intrinsic_i = inv_intrinsic[i:i + batchsize]
        real_img_i = real_img[i:i + batchsize]
        real_mask_i = real_mask[i:i + batchsize]
        fake_img_i, fake_mask_i, fine_weights, fine_depth = gen(pose_to_camera_i, pose_to_world_i,
                                                                bone_length_i, z, inv_intrinsic_i)
        loss_recon_i = mse(real_img_i, fake_img_i) + mse(real_mask_i, fake_mask_i)
        loss_kl_i = -0.5 * torch.sum(1 + torch.log(z_std) * 2 - z_mean ** 2 - z_std ** 2)

        (loss_recon_i + loss_kl_i).backward()
        loss_recon += loss_recon_i
        loss_kl += loss_kl_i

        fake_img.append(fake_img_i.detach())
    fake_img = torch.cat(fake_img)

    if rank == 0:
        if iter % 100 == 0:
            print(iter)
            write(iter, loss_recon, "loss_recon", writer)
            write(iter, loss_kl, "loss_kl", writer)

    torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
    gen_optimizer.step()
    enc_optimizer.step()

    # update discriminator
    gen_optimizer.zero_grad(set_to_none=True)
    enc_optimizer.zero_grad(set_to_none=True)

    return fake_img


def train_func(config, dataset, data_loader, rank, ddp=False, world_size=1):
    # TODO(xiao): move outside
    torch.backends.cudnn.benchmark = True

    out_dir = config.out_root
    out_name = config.out
    if rank == 0:
        writer = tbx.SummaryWriter(f"{out_dir}/runs/{out_name}")
        os.makedirs(f"{out_dir}/result/{out_name}", exist_ok=True)
        record_setting(f"{out_dir}/result/{out_name}")
    else:
        writer = None

    size = config.dataset.image_size
    num_iter = config.num_iter
    batchsize = config.dataset.bs
    n_accum_step = config.n_accum_step

    gen, enc = prepare_models(config.generator_params, dataset, size)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    gen = gen.cuda(n_gpu)
    enc = enc.cuda(n_gpu)

    if ddp:
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[n_gpu])
        enc = nn.parallel.DistributedDataParallel(enc, device_ids=[n_gpu])

    gen_lr = 1e-3

    gen_optimizer = optim.Adam(gen.parameters(), lr=gen_lr, betas=(0.9, 0.99))
    enc_optimizer = optim.Adam(enc.parameters(), lr=gen_lr / 10, betas=(0.9, 0.99))

    iter = 0
    start_time = time.time()

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")

            if ddp:
                gen_module = gen.module
                enc_module = enc.module
            else:
                gen_module = gen
                enc_module = enc

            gen_module.load_state_dict(snapshot["gen"], strict=False)
            enc_module.load_state_dict(snapshot["enc"])

            # gen_optimizer.load_state_dict(snapshot["gen_opt"])
            # enc_optimizer.load_state_dict(snapshot["dis_opt"])
            iter = snapshot["iteration"]
            # start_time = snapshot["start_time"]
            del snapshot
    init_iter = iter

    # for debug
    # with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA],
    #
    #         schedule=torch.profiler.schedule(
    #             wait=2,
    #             warmup=2,
    #             active=5),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #             '/data/unagi0/noguchi/D1/NARF_GAN/profile/tb_log'), ) as p:
    while iter < num_iter:
        for i, data in enumerate(data_loader):
            if (iter + 1) % 10 == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / (iter - init_iter + 1)} s/iter")
            gen.train()

            real_img = data["img"].cuda(non_blocking=True).float()
            real_mask = data["mask"].cuda(non_blocking=True).float()
            bone_mask = data["bone_mask"].cuda(non_blocking=True)
            pose_to_camera = data["pose_3d"].cuda(non_blocking=True)
            bone_length = data["bone_length"].cuda(non_blocking=True)
            pose_to_world = data["pose_3d_world"].cuda(non_blocking=True)
            intrinsic = data["intrinsics"].cuda(non_blocking=True)
            inv_intrinsic = torch.inverse(intrinsic)

            if real_img.shape[0] != batchsize or bone_mask.shape[0] != batchsize:  # drop last minibatch
                continue

            fake_img = train_step(iter, gen, pose_to_camera, pose_to_world, bone_length, inv_intrinsic,
                                  enc, gen_optimizer, enc_optimizer,
                                  rank, writer, real_img, real_mask, n_accum_step)
            # try:
            #     fake_img = train_step(iter, batchsize, gen, pose_to_camera, pose_to_world, bone_length, inv_intrinsic,
            #                           bone_loss_func, bone_mask, dis, ddp, world_size, gen_optimizer, dis_optimizer,
            #                           adv_loss_type, rank, writer, real_img, r1_loss_coef)
            # except:
            #     print("failed")
            #     torch.cuda.empty_cache()
            #     continue

            if rank == 0:
                if iter == 10:
                    with open(f"{out_dir}/result/{out_name}/iter_10_succeeded.txt", "w") as f:
                        f.write("ok")
                if iter % 200 == 0:
                    print(fake_img.shape)
                    save_img(fake_img, f"{out_dir}/result/{out_name}/rgb_{iter // 5000 * 5000}.png")
                    save_img(real_img, f"{out_dir}/result/{out_name}/real.png")
                    save_img(bone_mask, f"{out_dir}/result/{out_name}/bone_{iter // 5000 * 5000}.png")
                if (iter + 1) % 200 == 0:
                    if ddp:
                        gen_module = gen.module
                        enc_module = enc.module
                    else:
                        gen_module = gen
                        enc_module = enc
                    save_params = {"iteration": iter,
                                   "start_time": start_time,
                                   "gen": gen_module.state_dict(),
                                   "enc": enc_module.state_dict(),
                                   "gen_opt": gen_optimizer.state_dict(),
                                   "enc_opt": enc_optimizer.state_dict(),
                                   }
                    torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_latest.pth")
                    torch.save(save_params,
                               f"{out_dir}/result/{out_name}/snapshot_{(iter // 50000 + 1) * 50000}.pth")

            # torch.cuda.empty_cache()
            iter += 1
            # p.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/NARF_GAN/THUman/20210903.yml")
    parser.add_argument('--default_config', type=str, default="configs/NARF_VAE/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    train(train_func, config)
