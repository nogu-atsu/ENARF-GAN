import argparse
import os
import time
import warnings

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset import HumanDataset, HumanPoseDataset
from libraries.config import yaml_config
from libraries.custom_stylegan2.net import Discriminator
from libraries.gan.loss import adv_loss_dis, adv_loss_gen, d_r1_loss
from libraries.train_utils import record_command, write, save_img
from models.generator import TriNARFGenerator
from models.loss import nerf_patch_loss

warnings.filterwarnings('ignore')


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
    if dataset_name == "human_v2":
        img_dataset = HumanDataset(train_dataset_config, size=size, return_bone_params=False,
                                   just_cache=just_cache)

        pose_prior_root = train_dataset_config.pose_prior_root or train_dataset_config.data_root
        print("pose prior:", pose_prior_root)
        pose_dataset = HumanPoseDataset(size=size, data_root=pose_prior_root,
                                        just_cache=just_cache)
    else:
        assert False
    return img_dataset, pose_dataset


def create_dataloader(config_dataset):
    batch_size = config_dataset.bs
    shuffle = True
    drop_last = True
    num_workers = config_dataset.num_workers
    print("num_workers:", num_workers)

    img_dataset, pose_dataset = create_dataset(config_dataset)
    loader_img = DataLoader(img_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                            drop_last=drop_last)
    loader_pose = DataLoader(pose_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                             drop_last=drop_last)

    return (img_dataset, pose_dataset), (loader_img, loader_pose)


def prepare_models(gen_config, dis_config, pose_dataset, size):
    if gen_config.use_triplane:
        gen = TriNARFGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                               num_bone_param=pose_dataset.num_bone_param,
                               parent_id=pose_dataset.parents)
        gen.register_canonical_pose(pose_dataset.canonical_pose)
    else:
        raise ValueError("generator without triplane is not supported")
    dis = Discriminator(dis_config, size=size)
    return gen, dis


def loss(gen, dis, fake_img, fake_low_res_mask, bone_mask, background_ratio,
         bone_loss_func, gen_optimizer, dis_optimizer, adv_loss_type, ddp,
         world_size):
    loss_dict = {}
    loss_bone = bone_loss_func(fake_low_res_mask, bone_mask,
                               background_ratio) * config.loss.bone_guided_coef

    dis_fake = dis(fake_img, ddp, world_size)
    gen_optimizer.zero_grad(set_to_none=True)
    dis_optimizer.zero_grad(set_to_none=True)
    loss_adv_gen = adv_loss_gen(dis_fake, adv_loss_type, tmp=1)
    loss_gen = loss_adv_gen + loss_bone

    if config.loss.tri_plane_reg_coef > 0:
        loss_triplane = gen.nerf.buffers_tensors["tri_plane_feature"].square().mean()
        loss_gen += loss_triplane * config.loss.tri_plane_reg_coef

    loss_dict["adv_loss_gen"] = loss_adv_gen
    loss_dict["bone_loss"] = loss_bone
    return loss_gen, loss_dict


def train_step(iter, batchsize, gen, pose_to_camera, pose_to_world, bone_length, inv_intrinsic,
               bone_loss_func, bone_mask, dis, ddp, world_size, gen_optimizer, dis_optimizer,
               adv_loss_type, rank, writer, real_img, r1_loss_coef):
    n_accum_step = config.n_accum_step
    forward_bs = batchsize // n_accum_step
    fake_img = []
    dis.requires_grad_(False)
    for i in range(0, batchsize, forward_bs):
        # randomly sample latent
        z = torch.cuda.FloatTensor(forward_bs, config.generator_params.z_dim * 4).normal_()

        fake_img_i, fake_mask_i, fine_weights, fine_depth = gen(pose_to_camera[i:i + forward_bs],
                                                                pose_to_world[i:i + forward_bs],
                                                                bone_length[i:i + forward_bs], z,
                                                                inv_intrinsic[i:i + forward_bs])

        background_ratio = gen.background_ratio

        loss_gen, loss_dict = loss(gen, dis, fake_img_i, fake_mask_i,
                                   bone_mask[i:i + forward_bs], background_ratio, bone_loss_func, gen_optimizer,
                                   dis_optimizer, adv_loss_type, ddp, world_size)

        loss_gen.backward()

        fake_img.append(fake_img_i)

    fake_img = torch.cat(fake_img)

    gen_optimizer.step()

    if rank == 0:
        if iter % 100 == 0:
            print(iter)
            for k, v in loss_dict.items():
                write(iter, v, k, writer)

    torch.cuda.empty_cache()

    # update discriminator
    gen_optimizer.zero_grad(set_to_none=True)
    dis_optimizer.zero_grad(set_to_none=True)
    dis.requires_grad_(True)
    dis_fake = dis(fake_img.detach(), ddp, world_size)
    dis_real = dis(real_img, ddp, world_size)

    loss_dis = adv_loss_dis(dis_real, dis_fake, adv_loss_type)
    if rank == 0:
        if iter % 100 == 0:
            write(iter, loss_dis, "adv_loss_dis", writer)

    loss_dis.backward()
    dis_optimizer.step()

    if iter % 16 == 0:
        gen_optimizer.zero_grad(set_to_none=True)
        dis_optimizer.zero_grad(set_to_none=True)
        real_img.requires_grad = True
        torch.cuda.empty_cache()

        dis_real = dis(real_img, ddp, world_size)
        r1_loss = d_r1_loss(dis_real, real_img)
        if rank == 0:
            write(iter, r1_loss, "r1_reg", writer)

        (1 / 2 * r1_loss * 16 * r1_loss_coef + 0 * dis_real[
            0]).backward()  # 0 * dis_real[0] avoids zero grad
        dis_optimizer.step()
        torch.cuda.empty_cache()
    return fake_img


def train_func(config, datasets, data_loaders, rank, ddp=False, world_size=1):
    torch.backends.cudnn.benchmark = True

    out_dir = config.out_root
    out_name = config.out
    if rank == 0:
        writer = tbx.SummaryWriter(f"{out_dir}/runs/{out_name}")
        os.makedirs(f"{out_dir}/result/{out_name}", exist_ok=True)
        record_command(f"{out_dir}/result/{out_name}")
    else:
        writer = None

    size = config.dataset.image_size
    num_iter = config.num_iter
    batchsize = config.dataset.bs
    adv_loss_type = config.loss.adv_loss_type
    r1_loss_coef = config.loss.r1_loss_coef

    img_dataset, pose_dataset = datasets
    loader_img, loader_pose = data_loaders

    gen, dis = prepare_models(config.generator_params, config.discriminator_params, pose_dataset, size)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    gen = gen.cuda(n_gpu)
    dis = dis.cuda(n_gpu)

    if ddp:
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[n_gpu])
        dis = nn.parallel.DistributedDataParallel(dis, device_ids=[n_gpu])

    bone_loss_func = nerf_patch_loss

    gen_lr = 1e-3 * batchsize / 32
    dis_lr = 2e-3 * batchsize / 32

    gen_optimizer = optim.Adam(gen.parameters(), lr=gen_lr, betas=(0, 0.99))
    dis_optimizer = optim.Adam(dis.parameters(), lr=dis_lr, betas=(0, 0.99))

    iter = 0
    start_time = time.time()

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")

            if ddp:
                gen_module = gen.module
                dis_module = dis.module
            else:
                gen_module = gen
                dis_module = dis
            gen_module.load_state_dict(snapshot["gen"], strict=False)
            dis_module.load_state_dict(snapshot["dis"])

            # gen_optimizer.load_state_dict(snapshot["gen_opt"])
            # dis_optimizer.load_state_dict(snapshot["dis_opt"])
            iter = snapshot["iteration"]
            del snapshot
    init_iter = iter

    while iter < num_iter:
        for i, (img, pose) in enumerate(zip(loader_img, loader_pose)):
            if (iter + 1) % 10 == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / (iter - init_iter + 1)} s/iter")
            gen.train()
            dis.train()
            dis.requires_grad_(False)

            real_img = img["img"].cuda(non_blocking=True).float()
            bone_mask = pose["bone_mask"].cuda(non_blocking=True)
            pose_to_camera = pose["pose_to_camera"].cuda(non_blocking=True)
            bone_length = pose["bone_length"].cuda(non_blocking=True)
            pose_to_world = pose["pose_to_world"].cuda(non_blocking=True)
            intrinsic = pose["intrinsics"].cuda(non_blocking=True)
            inv_intrinsic = torch.inverse(intrinsic)

            if real_img.shape[0] != batchsize or bone_mask.shape[0] != batchsize:  # drop last minibatch
                continue

            # fake_img = train_step(iter, batchsize, gen, pose_to_camera, pose_to_world, bone_length, inv_intrinsic,
            #                       bone_loss_func, bone_mask, dis, ddp, world_size, gen_optimizer, dis_optimizer,
            #                       adv_loss_type, rank, writer, real_img, r1_loss_coef)
            try:
                fake_img = train_step(iter, batchsize, gen, pose_to_camera, pose_to_world, bone_length, inv_intrinsic,
                                      bone_loss_func, bone_mask, dis, ddp, world_size, gen_optimizer, dis_optimizer,
                                      adv_loss_type, rank, writer, real_img, r1_loss_coef)
            except:
                print("iteration skipped")
                torch.cuda.empty_cache()
                continue

            if rank == 0:
                if iter == 10:
                    with open(f"{out_dir}/result/{out_name}/iter_10_succeeded.txt", "w") as f:
                        f.write("ok")
                if iter % 50 == 0:
                    save_img(fake_img, f"{out_dir}/result/{out_name}/rgb_{iter // 5000 * 5000}.png")
                    save_img(real_img, f"{out_dir}/result/{out_name}/real.png")
                    save_img(bone_mask, f"{out_dir}/result/{out_name}/bone_{iter // 5000 * 5000}.png")
                if (iter + 1) % 200 == 0:
                    if ddp:
                        gen_module = gen.module
                        dis_module = dis.module
                    else:
                        gen_module = gen
                        dis_module = dis
                    save_params = {"iteration": iter,
                                   "start_time": start_time,
                                   "gen": gen_module.state_dict(),
                                   "dis": dis_module.state_dict(),
                                   "gen_opt": gen_optimizer.state_dict(),
                                   "dis_opt": dis_optimizer.state_dict(),
                                   }
                    torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_latest.pth")
                    torch.save(save_params,
                               f"{out_dir}/result/{out_name}/snapshot_{(iter // 50000 + 1) * 50000}.pth")

            torch.cuda.empty_cache()
            iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/enarfgan/AIST/enarfgan.yml")
    parser.add_argument('--default_config', type=str, default="configs/enarfgan/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    train(train_func, config)
