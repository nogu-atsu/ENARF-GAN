import argparse
import os
import time
import warnings

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.train_utils import record_setting, write
from utils.config import yaml_config
from utils.visualization_utils import save_img
from dataset import THUmanDataset, THUmanPoseDataset, HumanDataset, HumanPoseDataset
from models.loss import adv_loss_dis, adv_loss_gen, d_r1_loss, nerf_patch_loss, loss_dist_func
from models.net import NeRFNRGenerator, TriNeRFGenerator
from models.stylegan import Discriminator

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
    if dataset_name == "human":
        img_dataset = THUmanDataset(train_dataset_config, size=size, return_bone_params=False,
                                    just_cache=just_cache)

        pose_prior_root = train_dataset_config.pose_prior_root or train_dataset_config.data_root
        pose_prior_root2 = train_dataset_config.pose_prior_root2  # default is None
        pose1_ratio = train_dataset_config.pose1_ratio  # default is 1
        pose2_ratio = train_dataset_config.pose2_ratio  # default is 0

        print("pose prior:", pose_prior_root)
        pose_dataset = THUmanPoseDataset(size=size, data_root=pose_prior_root,
                                         just_cache=just_cache, data_root2=pose_prior_root2,
                                         data1_ratio=pose1_ratio, data2_ratio=pose2_ratio)
    elif dataset_name == "human_v2":
        # TODO mixed prior
        img_dataset = HumanDataset(train_dataset_config, size=size, return_bone_params=False,
                                   just_cache=just_cache)

        pose_prior_root = train_dataset_config.pose_prior_root or train_dataset_config.data_root
        print("pose prior:", pose_prior_root)
        pose_dataset = HumanPoseDataset(size=size, data_root=pose_prior_root,
                                        just_cache=just_cache)
    elif dataset_name == "human_tight":
        img_dataset = HumanDataset(train_dataset_config, size=size, return_bone_params=False,
                                   just_cache=just_cache)
        pose_prior_root = train_dataset_config.pose_prior_root or train_dataset_config.data_root
        pose_prior_root2 = train_dataset_config.pose_prior_root2  # default is None
        pose1_ratio = train_dataset_config.pose1_ratio  # default is 1
        pose2_ratio = train_dataset_config.pose2_ratio  # default is 0

        print("pose prior:", pose_prior_root)
        pose_dataset = THUmanPoseDataset(size=size, data_root=pose_prior_root,
                                         just_cache=just_cache, data_root2=pose_prior_root2,
                                         data1_ratio=pose1_ratio, data2_ratio=pose2_ratio, crop_algo="tight")

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
        gen = TriNeRFGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                               num_bone_param=pose_dataset.num_bone_param,
                               parent_id=pose_dataset.parents)
        gen.register_canonical_pose(pose_dataset.canonical_pose)
    else:
        gen = NeRFNRGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                              num_bone_param=pose_dataset.num_bone_param, parent_id=pose_dataset.parents)
    dis = Discriminator(dis_config, size=size)
    return gen, dis


def loss(gen, dis, batchsize, fake_img, fake_low_res_mask, fine_weights, fine_depth, bone_mask, background_ratio,
         bone_loss_func, gen_optimizer, dis_optimizer, adv_loss_type, ddp,
         world_size):
    loss_dict = {}
    print(config.loss.bone_guided_coef, background_ratio)
    loss_bone = bone_loss_func(fake_low_res_mask, bone_mask,
                               background_ratio) * config.loss.bone_guided_coef

    dis_fake = dis(fake_img, ddp, world_size)
    gen_optimizer.zero_grad(set_to_none=True)
    dis_optimizer.zero_grad(set_to_none=True)
    loss_adv_gen = adv_loss_gen(dis_fake, adv_loss_type, tmp=1)
    loss_gen = loss_adv_gen + loss_bone

    if config.loss.surface_reg_coef > 0:
        loss_dist = loss_dist_func(fine_weights, fine_depth)
        loss_gen += loss_dist * config.loss.surface_reg_coef
        loss_dict["loss_dist"] = loss_dist

    if config.loss.tri_plane_reg_coef > 0:
        loss_triplane = gen.nerf.buffers_tensors["tri_plane_feature"].square().mean()
        loss_gen += loss_triplane * config.loss.tri_plane_reg_coef

    if config.loss.tri_plane_mask_reg_coef > 0:
        mask = gen.nerf.buffers_tensors["tri_plane_feature"][:, 32 * 3:].clamp_min(-5)
        loss_triplane_mask = mask.mean() + mask.var(dim=0).mean() * 100
        # loss_triplane_mask = gen.nerf.buffers_tensors["tri_plane_feature"][:, 32 * 3:].var(dim=0).mean()
        loss_gen += loss_triplane_mask * config.loss.tri_plane_mask_reg_coef

    if config.loss.tri_plane_square_reg_coef > 0:
        mask = gen.nerf.buffers_tensors["tri_plane_feature"][:, 32 * 3:]  # (B, n_bone * 3, 256, 256)
        device = mask.device

        arange = torch.arange(256, device=device)
        pixel_location = (torch.stack(torch.meshgrid(arange, arange)[::-1]) + 0.5) / 128 - 1  # (2, 256, 256)

        canonical_joints = gen.nerf.canonical_joints[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        canonical_parent_joints = gen.nerf.canonical_parent_joints[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)

        joint_to_pixel = pixel_location - canonical_joints[:, :, None, None]  # (3n_bone, 2, 256, 256)
        bone_direction = F.normalize(canonical_parent_joints - canonical_joints, dim=1)  # (3n_bone, 2)
        joint_pixel_distance = joint_to_pixel.square().sum(dim=1)  # (3n_bone, 256, 256)
        pixel_bone_inner_product = (joint_to_pixel * bone_direction[:, :, None, None]).sum(dim=1)  # (3n_bone, 256, 256)
        bone_pixel_distance = joint_pixel_distance - pixel_bone_inner_product ** 2  # (3n_bone, 256, 256)
        parent_joint_pixel_distance = (pixel_location - canonical_parent_joints[:, :, None,
                                                        None]).square().sum(dim=1)  # (3n_bone, 256, 256)
        _bone_length = torch.norm(canonical_parent_joints - canonical_joints, dim=1)  # (3n_bone, )

        distance = torch.where(pixel_bone_inner_product >= 0, bone_pixel_distance, joint_pixel_distance)
        distance = torch.where(pixel_bone_inner_product <= _bone_length[:, None, None],
                               distance, parent_joint_pixel_distance)
        distance = distance - 4e-4
        distance = distance.masked_fill(distance < 0, -10)

        # # distance between part center and pixel
        # part_center = gen.nerf.canonical_pose[:, [0, 1, 1, 2, 2, 0], 3].reshape(-1, 2)  # (num_bone * 3, 2)
        # xy = pixel_location - part_center[:, :, None, None]
        # distance = xy.square().sum(dim=1)  # (num_bone * 3, 256, 256)

        tri_plane_square_reg = (torch.sigmoid(mask.clamp_max(6)) * distance).mean()
        loss_gen += tri_plane_square_reg * config.loss.tri_plane_square_reg_coef

    if config.loss.pseudo_mask_reg_coef > 0:
        fine_density = gen.nerf.buffers_tensors["fine_density"].detach()  # (B, 1, n)
        mask_weight = gen.nerf.buffers_tensors["mask_weight"]  # (B, n_bone, n)
        mask_prob = mask_weight.square().sum(dim=1) / (mask_weight.sum(dim=1) + 1e-2)
        pseudo_label = fine_density.squeeze(1) > 5
        loss_pseudo_mask = F.binary_cross_entropy(mask_prob, pseudo_label.float())
        loss_gen += loss_pseudo_mask * config.loss.pseudo_mask_reg_coef

    if config.loss.mask_derivative_reg_coef > 0:
        mask = gen.nerf.buffers_tensors["tri_plane_feature"][:, 32 * 3:]
        device = mask.device
        dx = F.conv2d(mask.reshape(-1, 1, 256, 256), torch.tensor([[-1, 0, 1],
                                                                   [-2, 0, 2],
                                                                   [-1, 0, 1]], dtype=torch.float32,
                                                                  device=device)[None, None])
        dy = F.conv2d(mask.reshape(-1, 1, 256, 256), torch.tensor([[-1, -2, -1],
                                                                   [0, 0, 0],
                                                                   [1, 2, 1]], dtype=torch.float32,
                                                                  device=device)[None, None])
        dx = dx.reshape(batchsize, -1, 3, 1, 254, 254)
        dy = dy.reshape(batchsize, -1, 3, 1, 254, 254)
        dxy = torch.cat([dx, dy], dim=3)
        dxy = F.normalize(dxy, dim=3)  # (B, num_bone, 3, 2, 254, 254)

        joint_location = gen.nerf.canonical_pose[:, [0, 1, 1, 2, 2, 0], 3].reshape(-1, 3, 2)  # (num_bone, 3, 2)
        arange = torch.arange(1, 255, device=device)
        pixel_location = (torch.stack(torch.meshgrid(arange, arange)[::-1]) + 0.5) / 128 - 1  # (2, 254, 254)
        xy = pixel_location - joint_location[:, :, :, None, None]
        xy = F.normalize(xy, dim=2)  # (num_bone, 3, 2, 254, 254)

        loss_mask_derivative_reg = (dxy * xy).clamp_min(-0.8).sum(dim=3).mean()
        loss_gen += loss_mask_derivative_reg * config.loss.mask_derivative_reg_coef
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

        loss_gen, loss_dict = loss(gen, dis, batchsize, fake_img_i, fake_mask_i, fine_weights, fine_depth,
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

    # torch.cuda.empty_cache()

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

        # mix_ratio = torch.rand(real_img.shape[0], device=real_img.device)[:, None, None, None]
        # mixed_img = real_img * mix_ratio + fake_img.detach() * (1 - mix_ratio)
        mixed_img = real_img

        dis_real = dis(mixed_img, ddp, world_size)
        r1_loss = d_r1_loss(dis_real, mixed_img)
        if rank == 0:
            write(iter, r1_loss, "r1_reg", writer)

        (1 / 2 * r1_loss * 16 * r1_loss_coef + 0 * dis_real[
            0]).backward()  # 0 * dis_real[0] avoids zero grad
        dis_optimizer.step()
        torch.cuda.empty_cache()
    return fake_img


def train_func(config, datasets, data_loaders, rank, ddp=False, world_size=1):
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
            # for k in list(snapshot["gen"].keys()):
            #     if "activate.bias" in k:
            #         snapshot["gen"][k[:-13] + "bias"] = snapshot["gen"][k].reshape(1, -1, 1, 1)
            #         del snapshot["gen"][k]
            gen_module.load_state_dict(snapshot["gen"], strict=False)
            dis_module.load_state_dict(snapshot["dis"])
            # gen.init_bg()
            # gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3, betas=(0, 0.99))

            # gen_optimizer.load_state_dict(snapshot["gen_opt"])
            # dis_optimizer.load_state_dict(snapshot["dis_opt"])
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

            fake_img = train_step(iter, batchsize, gen, pose_to_camera, pose_to_world, bone_length, inv_intrinsic,
                                  bone_loss_func, bone_mask, dis, ddp, world_size, gen_optimizer, dis_optimizer,
                                  adv_loss_type, rank, writer, real_img, r1_loss_coef)
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
                if iter % 50 == 0:
                    print(fake_img.shape)
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

            # torch.cuda.empty_cache()
            iter += 1
            # p.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/NARF_GAN/THUman/20210903.yml")
    parser.add_argument('--default_config', type=str, default="configs/NARF_GAN/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    train(train_func, config)
