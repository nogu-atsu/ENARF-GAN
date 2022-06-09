import argparse
import os
import time

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import THUmanDataset
from dependencies.NARF.pose_utils import rotate_pose_randomly
from dependencies.config import yaml_config
from dependencies.gan.loss import adv_loss_dis, adv_loss_gen, d_r1_loss
from dependencies.train_utils import record_setting, write
from models.misc import Encoder, PoseDiscriminator


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
        img_dataset = THUmanDataset(train_dataset_config, size=size, just_cache=just_cache)
        test_img_dataset = THUmanDataset(test_dataset_config, size=size, just_cache=just_cache,
                                         num_repeat_in_epoch=1)
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


def prepare_models(enc_config, img_dataset):
    enc = Encoder(enc_config, img_dataset.parents, img_dataset.cp.intrinsics)
    pdis = PoseDiscriminator(num_bone=img_dataset.num_bone)
    return enc, pdis


def evaluate(enc, test_loader):
    print("validation")
    mse = nn.MSELoss()
    enc.eval()
    loss_rotation = 0
    loss_translation = 0
    with torch.no_grad():
        for minibatch in tqdm(test_loader):
            real_img = minibatch["img"].cuda(non_blocking=True).float()
            pose_2d = minibatch["pose_2d"].cuda(non_blocking=True).float()
            pose_3d_gt = minibatch["pose_3d"].cuda(non_blocking=True).float()
            bs = real_img.shape[0]

            pose_3d, z, bone_length, intrinsic = enc(real_img, pose_2d)
            scaled_pose_3d_gt = enc.scale_pose(pose_3d_gt[:, :, :3, 3:])
            loss_rotation += mse(pose_3d[:, :, :3, :3], pose_3d_gt[:, :, :3, :3]) * bs
            loss_translation += mse(pose_3d[:, :, :3, 3:], scaled_pose_3d_gt) * bs

    loss_rotation = loss_rotation / len(test_loader.dataset)
    loss_translation = loss_translation / len(test_loader.dataset)

    loss_dict = {}
    loss_dict["loss_rotation_val"] = loss_rotation
    loss_dict["loss_translation_val"] = loss_translation

    enc.train()

    return loss_dict


def train_step(enc, pdis, real_img, pose_2d, adv_loss_type, enc_optimizer, pdis_optimizer):
    enc_optimizer.zero_grad()
    pdis_optimizer.zero_grad()

    # reconstruction
    pose_3d, z, bone_length, intrinsic = enc(real_img, pose_2d)

    # pose discriminator
    pose_3d_rotated = rotate_pose_randomly(pose_3d)
    fake_pose2d = torch.matmul(intrinsic, pose_3d_rotated[:, :, :3, 3:])
    fake_pose2d = fake_pose2d[:, :, :2, 0] / fake_pose2d[:, :, 2:, 0]

    pdis_fake = pdis(fake_pose2d)
    loss_pose_gen = adv_loss_gen(pdis_fake, adv_loss_type)

    loss_pose_gen.backward()
    enc_optimizer.step()

    loss_dict = {}
    loss_dict["loss_pose_gen"] = loss_pose_gen.item()

    enc_optimizer.zero_grad()
    pdis_optimizer.zero_grad()
    pdis_fake = pdis(fake_pose2d.detach())
    pdis_real = pdis(pose_2d)
    loss_pose_dis = adv_loss_dis(pdis_real, pdis_fake, adv_loss_type)

    loss_pose_dis.backward()
    pdis_optimizer.step()

    loss_dict["loss_pose_dis"] = loss_pose_dis.item()

    return loss_dict


def r1_regularization_step(pose_2d, pdis, pdis_optimizer, r1_loss_coef):
    pdis_optimizer.zero_grad()

    pose_2d.requires_grad = True
    pdis_real = pdis(pose_2d)
    r1_loss = d_r1_loss(pdis_real, pose_2d)

    (1 / 2 * r1_loss * 16 * r1_loss_coef).backward()  # 0 * dis_real[0] avoids zero grad
    pdis_optimizer.step()
    return r1_loss.item()


def train_func(config, datasets, data_loaders, rank, ddp=False, world_size=1):
    # TODO(xiao): move outside
    torch.backends.cudnn.benchmark = True

    out_dir = config.out_root
    out_name = config.out
    if rank == 0:
        writer = tbx.SummaryWriter(f"{out_dir}/runs/{out_name}")
        os.makedirs(f"{out_dir}/result/{out_name}", exist_ok=True)
        record_setting(f"{out_dir}/result/{out_name}")

    adv_loss_type = config.loss.adv_loss_type
    r1_loss_coef = config.loss.r1_loss_coef
    num_iter = config.num_iter

    img_dataset, test_img_dataset = datasets
    loader_img, test_loader_img = data_loaders

    enc, pdis = prepare_models(config.encoder_params, img_dataset)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    enc = enc.cuda(n_gpu)
    pdis = pdis.cuda(n_gpu)

    if ddp:
        enc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(enc)
        enc = nn.parallel.DistributedDataParallel(enc, device_ids=[n_gpu])
        pdis = nn.parallel.DistributedDataParallel(pdis, device_ids=[n_gpu])

    enc_optimizer = optim.Adam(enc.parameters(), lr=5e-4, betas=(0, 0.99))
    pdis_optimizer = optim.Adam(pdis.parameters(), lr=5e-4, betas=(0, 0.99))

    iter = 0
    start_time = time.time()

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")

            if ddp:
                enc_module = enc.module
                pdis_module = pdis.module
            else:
                enc_module = enc
                pdis_module = pdis
            enc_module.load_state_dict(snapshot["enc"], strict=True)
            pdis_module.load_state_dict(snapshot["pdis"], strict=True)
            enc_optimizer.load_state_dict(snapshot["enc_opt"])
            pdis_optimizer.load_state_dict(snapshot["pdis_opt"])
            iter = snapshot["iteration"]
            start_time = snapshot["start_time"]
            del snapshot

    while iter < num_iter:
        for i, minibatch in enumerate(loader_img):
            if (iter + 1) % 10 == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / iter} s/iter")
            enc.train()
            pdis.train()

            real_img = minibatch["img"].cuda(non_blocking=True).float()
            pose_2d = minibatch["pose_2d"].cuda(non_blocking=True).float()

            loss_dict = train_step(enc, pdis, real_img, pose_2d, adv_loss_type,
                                   enc_optimizer, pdis_optimizer)

            if rank == 0:
                if iter % 100 == 0:
                    print(iter)
                    for k, v in loss_dict.items():
                        write(iter, v, k, writer, True)

            if iter % 16 == 0:
                r1_loss = r1_regularization_step(pose_2d, pdis, pdis_optimizer, r1_loss_coef)

                if rank == 0:
                    if iter % 80 == 0:
                        write(iter, r1_loss, "r1_reg", writer, True)

            if rank == 0:
                if iter == 10:
                    with open(f"{out_dir}/result/{out_name}/iter_10_succeeded.txt", "w") as f:
                        f.write("ok")
                if (iter + 1) % 1000 == 0:
                    loss_dict_val = evaluate(enc, test_loader_img)
                    for k, v in loss_dict_val.items():
                        write(iter, v, k, writer, True)

                if (iter + 1) % 200 == 0:
                    if ddp:
                        enc_module = enc.module
                        pdis_module = pdis.module
                    else:
                        enc_module = enc
                        pdis_module = pdis
                    save_params = {"iteration": iter,
                                   "start_time": start_time,
                                   "enc": enc_module.state_dict(),
                                   "pdis": pdis_module.state_dict(),
                                   "enc_opt": enc_optimizer.state_dict(),
                                   "pdis_opt": pdis_optimizer.state_dict()}
                    print(f"{out_dir}/result/{out_name}/snapshot_latest.pth")
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
