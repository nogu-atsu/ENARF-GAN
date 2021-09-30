import argparse
import os
import time

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import THUmanDataset
from NARF.utils import record_setting, yaml_config, write
from NARF.visualization_utils import save_img
from models.loss import adv_loss_dis, adv_loss_gen, d_r1_loss, nerf_patch_loss
from models.net import NeRFNRGenerator, Encoder, PoseDiscriminator
from models.stylegan import Discriminator
from utils.mask_utils import create_bone_mask
from utils.rotation_utils import rotate_pose_randomly


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


def prepare_models(gen_config, dis_config, img_dataset, size):
    enc = Encoder(img_dataset.parents, img_dataset.cp.intrinsics)
    gen = NeRFNRGenerator(gen_config, size, img_dataset.cp.intrinsics, num_bone=img_dataset.num_bone,
                          num_bone_param=img_dataset.num_bone_param)
    dis = Discriminator(dis_config, size=size)
    pdis = PoseDiscriminator(num_bone=img_dataset.num_bone)
    return enc, gen, dis, pdis


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
    batchsize = config.dataset.bs
    adv_loss_type = config.loss.adv_loss_type
    r1_loss_coef = config.loss.r1_loss_coef

    img_dataset, = datasets
    loader_img, = data_loaders

    enc, gen, dis, pdis = prepare_models(config.generator_params, config.discriminator_params, img_dataset, size)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    enc = enc.cuda(n_gpu)
    gen = gen.cuda(n_gpu)
    dis = dis.cuda(n_gpu)
    pdis = pdis.cuda(n_gpu)

    mse = nn.MSELoss()

    if ddp:
        enc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(enc)
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        enc = nn.parallel.DistributedDataParallel(enc, device_ids=[n_gpu])
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[n_gpu])
        dis = nn.parallel.DistributedDataParallel(dis, device_ids=[n_gpu])
        pdis = nn.parallel.DistributedDataParallel(pdis, device_ids=[n_gpu])

    bone_loss_func = nerf_patch_loss

    enc_optimizer = optim.Adam(enc.parameters(), lr=5e-4, betas=(0, 0.99))
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3, betas=(0, 0.99))
    dis_optimizer = optim.Adam(dis.parameters(), lr=2e-3, betas=(0, 0.99))
    pdis_optimizer = optim.Adam(pdis.parameters(), lr=5e-4, betas=(0, 0.99))

    iter = 0
    start_time = time.time()

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")

            if ddp:
                enc_module = enc.module
                gen_module = gen.module
                dis_module = dis.module
                pdis_module = pdis.module
            else:
                enc_module = enc
                gen_module = gen
                dis_module = dis
                pdis_module = pdis
            enc_module.load_state_dict(snapshot["enc"], strict=True)
            gen_module.load_state_dict(snapshot["gen"], strict=True)
            dis_module.load_state_dict(snapshot["dis"])
            pdis_module.load_state_dict(snapshot["pdis"])
            enc_optimizer.load_state_dict(snapshot["enc_opt"])
            gen_optimizer.load_state_dict(snapshot["gen_opt"])
            dis_optimizer.load_state_dict(snapshot["dis_opt"])
            pdis_optimizer.load_state_dict(snapshot["pdis_opt"])
            iter = snapshot["iteration"]
            start_time = snapshot["start_time"]
            del snapshot

    while iter < num_iter:
        for i, minibatch in enumerate(loader_img):
            if (iter + 1) % 10 == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / iter} s/iter")
            enc.train()
            gen.train()
            dis.train()
            pdis.train()

            real_img = minibatch["img"].cuda(non_blocking=True).float()
            pose_2d = minibatch["pose_2d"].cuda(non_blocking=True).float()

            enc_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()
            pdis_optimizer.zero_grad()

            # reconstruction
            pose_3d, z, bone_length, intrinsic = enc(real_img,
                                                     pose_2d)  # (B, n_parts, 4, 4), (B, z_dim*4), (B, n_parts)

            fake_img, fake_low_res_mask = torch.utils.checkpoint.checkpoint(gen, pose_3d, None, bone_length, z)
            bone_mask = create_bone_mask(img_dataset.parents, pose_3d, size, intrinsic)

            loss_recon = mse(real_img, fake_img)

            # rotated images
            pose_3d_rotated = rotate_pose_randomly(pose_3d)
            fake_img_rotated, fake_low_res_mask_rotated = torch.utils.checkpoint.checkpoint(gen, pose_3d_rotated,
                                                                                            None, bone_length, z)
            bone_mask_rotated = create_bone_mask(img_dataset.parents, pose_3d_rotated, size, intrinsic)

            loss_bone = (bone_loss_func(fake_low_res_mask, bone_mask) +
                         bone_loss_func(fake_low_res_mask_rotated, bone_mask_rotated)) * config.loss.bone_guided_coef

            dis_fake = dis(fake_img, ddp, world_size)
            loss_adv_gen = adv_loss_gen(dis_fake, adv_loss_type, tmp=1)
            loss_gen = loss_adv_gen + loss_bone + loss_recon

            if rank == 0:
                print(iter)
                write(iter, loss_adv_gen, "adv_loss_gen", writer)
                write(iter, loss_bone, "bone_loss", writer)

            loss_gen.backward()
            gen_optimizer.step()
            enc_optimizer.step()

            torch.cuda.empty_cache()

            # update discriminator
            # image discriminator
            dis_fake = dis(fake_img_rotated.detach(), ddp, world_size)
            dis_real = dis(real_img, ddp, world_size)

            # pose discriminator
            fake_pose2d = torch.matmul(intrinsic, pose_3d_rotated[:, :, :3, 3:])
            fake_pose2d = fake_pose2d[:, :, :2, 0] / fake_pose2d[:, :, 2:, 0]
            pdis_fake = pdis(fake_pose2d.detach())
            pdis_real = pdis(pose_2d)

            loss_dis = adv_loss_dis(dis_real, dis_fake, adv_loss_type) + \
                       adv_loss_dis(pdis_real, pdis_fake, adv_loss_type)

            if rank == 0:
                write(iter, loss_dis, "adv_loss_dis", writer)

            enc_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()
            pdis_optimizer.zero_grad()
            loss_dis.backward()
            dis_optimizer.step()
            pdis_optimizer.step()

            if iter % 16 == 0:
                real_img.requires_grad = True
                pose_2d.requires_grad = True
                dis_real = dis(real_img, ddp, world_size)
                pdis_real = pdis(pose_2d)
                r1_loss = d_r1_loss(dis_real, real_img) + d_r1_loss(pdis_real, pose_2d)
                if rank == 0:
                    write(iter, r1_loss, "r1_reg", writer)

                enc_optimizer.zero_grad()
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()
                pdis_optimizer.zero_grad()
                (1 / 2 * r1_loss * 16 * r1_loss_coef + 0 * dis_real[0]).backward()  # 0 * dis_real[0] avoids zero grad
                dis_optimizer.step()
                pdis_optimizer.step()
            if rank == 0:
                if iter == 10:
                    with open(f"{out_dir}/result/{out_name}/iter_10_succeeded.txt", "w") as f:
                        f.write("ok")
                if iter % 50 == 0:
                    print(fake_img.shape)
                    save_img(fake_img, f"{out_dir}/result/{out_name}/rgb_{iter // 5000 * 5000}.png")
                    save_img(fake_img_rotated, f"{out_dir}/result/{out_name}/rgb_rotated_{iter // 5000 * 5000}.png")
                    save_img(real_img, f"{out_dir}/result/{out_name}/real.png")
                    save_img(bone_mask, f"{out_dir}/result/{out_name}/bone_{iter // 5000 * 5000}.png")
                    save_img(bone_mask_rotated, f"{out_dir}/result/{out_name}/bone_rotated_{iter // 5000 * 5000}.png")
                if (iter + 1) % 200 == 0:
                    if ddp:
                        enc_module = enc.module
                        gen_module = gen.module
                        dis_module = dis.module
                    else:
                        enc_module = enc
                        gen_module = gen
                        dis_module = dis
                    save_params = {"iteration": iter,
                                   "start_time": start_time,
                                   "enc": enc_module.state_dict(),
                                   "gen": gen_module.state_dict(),
                                   "dis": dis_module.state_dict(),
                                   "pdis": dis_module.state_dict(),
                                   "enc_opt": enc_optimizer.state_dict(),
                                   "gen_opt": gen_optimizer.state_dict(),
                                   "dis_opt": dis_optimizer.state_dict(),
                                   "pdis_opt": pdis_optimizer.state_dict(),
                                   }
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
