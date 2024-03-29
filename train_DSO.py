import argparse
import json
import os
import time
import warnings

import cv2
import numpy as np
import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import SSODataset
from libraries.NeRF.loss import PhotometricLoss
from libraries.config import yaml_config
from libraries.metrics import ssim, psnr, lpips, neural_actor_lpips
from libraries.train_utils import all_reduce
from libraries.train_utils import write
from models.generator import DSONARFGenerator

warnings.filterwarnings('ignore')


def train(config, validation=False):
    if validation:
        dataset, data_loader = create_dataloader(config.dataset)
        validation_func(config, dataset, data_loader, rank=0, ddp=False)
    else:
        dataset, data_loader = create_dataloader(config.dataset)
        train_func(config, dataset, data_loader, rank=0, ddp=False)


def create_dataloader(config_dataset):
    batch_size = config_dataset.bs
    shuffle = True
    drop_last = True
    num_workers = config_dataset.num_workers

    dataset_train, datasets_val = create_dataset(config_dataset)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                              drop_last=drop_last)
    val_loaders = {key: DataLoader(datasets_val[key], batch_size=1, num_workers=num_workers, shuffle=False,
                                   drop_last=False) for key in datasets_val.keys()}
    return (dataset_train, datasets_val), (train_loader, val_loaders)


def cache_dataset(config_dataset):
    create_dataset(config_dataset, just_cache=True)


def create_dataset(config_dataset, just_cache=False):
    size = config_dataset.image_size

    train_dataset_config = config_dataset.train
    val_dataset_config = config_dataset.val

    print("loading datasets")
    dataset_train = SSODataset(train_dataset_config, size=size, return_bone_params=True,
                               return_bone_mask=False, random_background=False, just_cache=just_cache,
                               load_camera_intrinsics=True, return_mask=True)
    datasets_val = dict()
    for key in val_dataset_config.keys():
        if val_dataset_config[key].data_root is not None:
            datasets_val[key] = SSODataset(val_dataset_config[key], size=size, return_bone_params=True,
                                           return_bone_mask=False, random_background=False,
                                           num_repeat_in_epoch=1, just_cache=just_cache,
                                           load_camera_intrinsics=True, return_mask=True)

    return dataset_train, datasets_val


def validate(gen, val_loaders, config, ddp=False, metric=["SSIM", "PSNR"], iter=0, num_data=None,
             crop=False):
    mse = nn.MSELoss()

    size = config.dataset.image_size
    bg_color = config.dataset.bg_color

    loss = dict()

    if "NeuralActor" in config.out:
        lpips_func = neural_actor_lpips
        print("using neural actor lpips")
    else:
        lpips_func = lpips
        print("using lpips")
    loss_func = {"L2": mse, "SSIM": ssim, "PSNR": psnr, "LPIPS": lpips_func}
    for key, val_loader in val_loaders.items():
        if num_data != 1 and key == "train":
            continue
        _num_data = len(val_loader.dataset) if num_data is None else min(num_data, len(val_loader.dataset))
        num_data_all = len(val_loader.dataset) if _num_data == 1 else _num_data

        val_loss_color = 0
        val_loss_mask = 0
        val_loss_color_metric = {met: 0 for met in metric}
        for i, data in tqdm(enumerate(val_loader), total=num_data_all):
            gen.eval()
            with torch.no_grad():
                batch = {key: val.cuda(non_blocking=True).float() for key, val in data.items()}

                img = batch["img"]
                mask = batch["mask"]
                pose_to_camera = batch["pose_3d"]
                frame_time = batch["frame_time"]
                bone_length = batch["bone_length"]
                camera_rotation = batch["camera_rotation"]
                intrinsic = batch["intrinsics"]
                inv_intrinsic = torch.inverse(intrinsic)

                # bbox
                if crop:
                    x_ = torch.where(mask[0].any(dim=0))[0]
                    if len(x_) == 0:
                        print("no mask")
                        continue
                    x_min, x_max = x_[0], x_[-1]
                    y_ = torch.where(mask[0].any(dim=1))[0]
                    if len(y_) == 0:
                        print("no mask")
                        continue
                    y_min, y_max = y_[0], y_[-1]
                    bbox = (x_min, y_min, x_max, y_max)

                    mask = mask[:, y_min:y_max, x_min:x_max]
                    img = img[:, :, y_min:y_max, x_min:x_max]
                else:
                    bbox = None

                gen_color, gen_mask, _ = gen.render_entire_img(pose_to_camera, inv_intrinsic, frame_time,
                                                               bone_length, camera_rotation, size, bbox=bbox)

                if torch.isnan(gen_color).any():
                    print("NaN is detected")
                gen_color = gen_color[None]
                gen_mask = gen_mask[None]
                gen_color = gen_color + bg_color * (1 - gen_mask)

                val_loss_mask += loss_func["L2"](mask, gen_mask).item()
                val_loss_color += loss_func["L2"](img, gen_color).item()

                for met in metric:
                    val_loss_color_metric[met] += loss_func[met](img, gen_color).item()

            if i == _num_data - 1:
                # save image
                gen_color = torch.cat([gen_color, img], dim=-1)
                gen_color = gen_color.cpu().numpy()[0].transpose(1, 2, 0) * 127.5 + 127.5
                gen_color = np.clip(gen_color, 0, 255)[:, :, ::-1]
                out_dir = config.out_root
                out_name = config.out
                cv2.imwrite(f"{out_dir}/result/{out_name}/{key}_{iter // 5000 * 5000}.png", gen_color)
            if i == _num_data - 1:
                break
        if ddp:
            val_loss_mask = all_reduce(val_loss_mask)
            val_loss_color = all_reduce(val_loss_color)

            for met in metric:
                val_loss_color_metric[met] = all_reduce(val_loss_color_metric[met])

        loss[key] = {"color": val_loss_color / num_data_all,
                     "mask": val_loss_mask / num_data_all}
        for met in metric:
            loss[key][f"color_{met}"] = val_loss_color_metric[met] / num_data_all

    return loss


def train_func(config, dataset, data_loader, rank, ddp=False, world_size=1):
    torch.backends.cudnn.benchmark = True

    out_dir = config.out_root
    out_name = config.out
    if rank == 0:
        writer = tbx.SummaryWriter(f"{out_dir}/runs/{out_name}")
        os.makedirs(f"{out_dir}/result/{out_name}", exist_ok=True)

    size = config.dataset.image_size
    num_iter = config.num_iter

    dataset = dataset[0]
    num_bone = dataset.num_bone

    gen = DSONARFGenerator(config.generator_params, size, num_bone,
                           parent_id=dataset.parents, num_bone_param=dataset.num_bone_param)
    gen.register_canonical_pose(dataset.canonical_pose)

    loss_func = PhotometricLoss(config.loss)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    gen = gen.cuda(n_gpu)

    if ddp:
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[n_gpu])

    gen_optimizer = optim.Adam(gen.parameters(), lr=config.lr, betas=(0.9, 0.99))

    if config.scheduler_gamma < 1:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, config.scheduler_gamma)

    start_time = time.time()
    iter = 0

    if config.resume or config.resume_latest:
        print(config.resume)
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")
            if ddp:
                gen_module = gen.module
            else:
                gen_module = gen

            for k, v in gen_module.named_parameters():
                if k not in snapshot["gen"]:
                    snapshot["gen"][k] = v.data.clone()

            for k, v in gen_module.named_buffers():
                if k not in snapshot["gen"]:
                    snapshot["gen"][k] = v.data.clone()

            gen_module.load_state_dict(snapshot["gen"], strict=True)
            # gen_optimizer.load_state_dict(snapshot["gen_opt"])
            iter = snapshot["iteration"]
            start_time = snapshot["start_time"]
            del snapshot

    train_loader, val_loaders = data_loader

    train_loss_color = 0
    train_loss_mask = 0

    accumulated_train_time = 0
    log = {}

    train_start = time.time()

    val_interval = config.val_interval
    print_interval = config.print_interval
    tensorboard_interval = config.tensorboard_interval
    save_interval = config.save_interval
    while iter < num_iter:
        for i, data in enumerate(train_loader):
            if (iter + 1) % print_interval == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / iter} s/iter")
            gen.train()
            batch = {key: val.cuda(non_blocking=True).float() for key, val in data.items()}
            img = batch["img"]
            mask = batch["mask"]
            pose_to_camera = batch["pose_3d"]
            frame_time = batch["frame_time"]
            bone_length = batch["bone_length"]
            camera_rotation = batch["camera_rotation"]
            intrinsic = batch["intrinsics"]
            inv_intrinsic = torch.inverse(intrinsic)

            gen_optimizer.zero_grad()
            # generate image (sparse sample)
            nerf_color, nerf_mask, grid = gen(pose_to_camera, camera_rotation, mask, frame_time,
                                              bone_length, inv_intrinsic, background=config.dataset.bg_color)
            loss_color, loss_mask = loss_func(grid, nerf_color, nerf_mask, img, mask)

            loss = loss_color + loss_mask

            # accumulate train loss
            train_loss_color += loss_color.item() * config.dataset.bs
            train_loss_mask += loss_mask.item() * config.dataset.bs

            if (iter + 1) % tensorboard_interval == 0 and rank == 0:  # tensorboard
                write(iter, loss, "gen", writer)
            loss.backward()

            gen_optimizer.step()

            if config.scheduler_gamma < 1:
                scheduler.step()
            # torch.cuda.empty_cache()

            if (iter + 1) % save_interval == 0 and rank == 0:
                if ddp:
                    gen_module = gen.module
                else:
                    gen_module = gen
                save_params = {"iteration": iter,
                               "start_time": start_time,
                               "gen": gen_module.state_dict(),
                               "gen_opt": gen_optimizer.state_dict(),
                               }
                torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_latest.pth")
                torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_{(iter // 50000 + 1) * 50000}.pth")
            if (iter + 1) % val_interval == 0:
                # add train time
                accumulated_train_time += time.time() - train_start

                val_loss = validate(gen, val_loaders, config, ddp, iter=iter, num_data=1)
                torch.cuda.empty_cache()

                if ddp:
                    train_loss_color = all_reduce(train_loss_color)
                    train_loss_mask = all_reduce(train_loss_mask)

                train_loss_color = train_loss_color / (val_interval * world_size * config.dataset.bs)
                train_loss_mask = train_loss_mask / (val_interval * world_size * config.dataset.bs)

                # write log
                log_ = {"accumulated_train_time": accumulated_train_time,
                        "train_loss_color": train_loss_color,
                        "train_loss_mask": train_loss_mask}
                for key in val_loss.keys():
                    for metric in val_loss[key].keys():
                        log_[f"val_loss_{key}_{metric}"] = val_loss[key][metric]

                log[iter + 1] = log_

                if rank == 0:
                    with open(f"{out_dir}/result/{out_name}/log.json", "w") as f:
                        json.dump(log, f)

                # initialize train loss
                train_loss_color = 0
                train_loss_mask = 0

                train_start = time.time()

            iter += 1


def validation_func(config, dataset, data_loader, rank, ddp=False):
    out_dir = config.out_root
    out_name = config.out
    size = config.dataset.image_size

    dataset = dataset[0]
    num_bone = dataset.num_bone

    gen = DSONARFGenerator(config.generator_params, size, num_bone,
                           parent_id=dataset.parents, num_bone_param=dataset.num_bone_param)
    gen.register_canonical_pose(dataset.canonical_pose)

    torch.cuda.set_device(rank)
    gen.cuda(rank)

    if ddp:
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[rank])

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")
            if ddp:
                gen_module = gen.module
            else:
                gen_module = gen
            gen_module.load_state_dict(snapshot["gen"], strict=False)
            del snapshot
    else:
        assert False, "Please load a pretrained model"

    _, val_loaders = data_loader

    val_loss = validate(gen, val_loaders, dataset, config, ddp, metric=["PSNR", "SSIM", "LPIPS"], crop=True)
    torch.cuda.empty_cache()
    # write log
    if rank == 0:
        with open(f"{out_dir}/result/{out_name}/val_metrics.json", "w") as f:
            json.dump(val_loss, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/DSO/ZJU/313_denarf.yml")
    parser.add_argument('--default_config', type=str, default="configs/DSO/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--validation', action="store_true")

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    train(config, args.validation)
