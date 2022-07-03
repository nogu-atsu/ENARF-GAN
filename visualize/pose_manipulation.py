import argparse
import os
import random
import sys
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(".")
from dataset.dataset import HumanPoseDataset
from models.generator import TriNARFGenerator
from dependencies.config import yaml_config
from dependencies.NARF.pose_utils import rotate_pose_by_angle, rotate_mesh_by_angle
from dependencies.NARF.mesh_rendering import render_mesh_

warnings.filterwarnings('ignore')


def render_img_and_mesh(gen, seed, z, l, j, w, intri, inv_intri, meshes):
    with torch.no_grad():
        torch.manual_seed(seed)
        fake_img, fake_low_res_mask, _, _ = gen(j, w, l, z=z, inv_intrinsics=inv_intri,
                                                truncation_psi=truncation_psi,
                                                black_bg_if_possible=whiten_bg)
        mesh_img = render_mesh_(meshes, intri, gen.size, render_size)

    if whiten_bg:
        fake_img = fake_img + 2 * (1 - fake_low_res_mask)

    out = fake_img.cpu().numpy()[0].transpose(1, 2, 0)
    out = out * 127.5 + 127.5
    out = np.clip(out, 0, 255).astype("uint8")
    out = cv2.resize(out, (mesh_img.shape[1], mesh_img.shape[0]))

    out = np.concatenate([out, mesh_img], axis=1)
    return out


def sample_mesh(gen, sequence_dataset, i, z):
    minibatch = sequence_dataset[i]
    minibatch = {k: torch.tensor(v).cuda() for k, v in minibatch.items()}
    j = minibatch["pose_to_camera"][None]
    w = minibatch["pose_to_world"][None]
    l = minibatch["bone_length"][None]
    intri = minibatch["intrinsics"][None]
    inv_intri = torch.inverse(intri)
    # eps = i / (num_batch - 1)
    # z = z0 * eps + z1 * (1 - eps)
    with torch.no_grad():
        meshes = gen.create_mesh(j.clone(), z, l, mesh_th=mesh_th, truncation_psi=truncation_psi)
    return j, w, l, intri, inv_intri, meshes


def main(mode: str = "pose_manipulation"):
    config = yaml_config(args.config, args.default_config)

    size = config.dataset.image_size

    sequence_data_root = "/data/unagi0/noguchi/dataset/aist++/sequence_cache/"
    sequence_dataset = HumanPoseDataset(size=128, data_root=f"{sequence_data_root}/{1:0>2d}")

    gen_config = config.generator_params

    if gen_config.use_triplane:
        gen = TriNARFGenerator(gen_config, size, num_bone=sequence_dataset.num_bone,
                               num_bone_param=sequence_dataset.num_bone_param,
                               parent_id=sequence_dataset.parents,
                               black_background=False)
        gen.register_canonical_pose(sequence_dataset.canonical_pose)
        gen.to("cuda")
    else:
        assert False

    iteration = args.iteration
    if os.path.exists(f"{config.out_root}/result/{config.out}/snapshot_{iteration}.pth"):
        state_dict = torch.load(f"{config.out_root}/result/{config.out}/snapshot_{iteration}.pth")["gen"]
        # for k in list(state_dict.keys()):
        #     if "activate.bias" in k:
        #         state_dict[k[:-13]+"bias"] = state_dict[k].reshape(1, -1, 1, 1)
        #         del state_dict[k]
        gen.load_state_dict(state_dict, strict=False)
    else:
        print("model not loaded")
    gen.train()
    num_batch = 30
    gen.eval()

    for img_id in args.person_id:
        nerf_imgs = []
        z = torch.cuda.FloatTensor(1, config.generator_params.z_dim * 4).normal_()
        sequence_dataset = HumanPoseDataset(size=128, data_root=f"{sequence_data_root}/{img_id:0>2d}")
        if mode == "pose_manipulation":
            seed = random.randint(0, 10000)
            for i in tqdm(range(num_batch)):
                j, w, l, intri, inv_intri, meshes = sample_mesh(gen, sequence_dataset, i * 4 + 100, z)
                out = render_img_and_mesh(gen, seed, z, l, j, w, intri, inv_intri, meshes)

                nerf_imgs.append(out)
        elif mode == "rotation":
            i = random.randint(0, len(sequence_dataset) - 1)
            sample_mesh(gen, sequence_dataset, i, z)
            j, w, l, intri, inv_intri, meshes = sample_mesh(gen, sequence_dataset, i, z)

            seed = random.randint(0, 10000)
            for i in tqdm(range(num_batch)):
                with torch.no_grad():
                    j_ = rotate_pose_by_angle(j.clone(), torch.tensor([2 * np.pi * i / num_batch]).cuda())
                    meshes_i = rotate_mesh_by_angle(j.clone(), meshes, torch.tensor([2 * np.pi * i / num_batch]).cuda())

                out = render_img_and_mesh(gen, seed, z, l, j_, w, intri, inv_intri, meshes_i)
                nerf_imgs.append(out)
        elif mode == "appearance_interpolation":
            seed = random.randint(0, 10000)

            t = random.randint(0, len(sequence_dataset) - 1)
            zs = torch.cuda.FloatTensor(3, config.generator_params.z_dim * 4).normal_()
            for i in tqdm(range(num_batch)):
                z_idx_left = i // (num_batch // len(zs))
                z_idx_right = (z_idx_left + 1) % len(zs)
                eps = (i % (num_batch // len(zs))) / (num_batch // len(zs))
                z_left = zs[z_idx_left][None]
                z_right = zs[z_idx_right][None]
                z_ = z_left * (1 - eps) + z_right * eps
                j, w, l, intri, inv_intri, meshes = sample_mesh(gen, sequence_dataset, t, z_)
                out = render_img_and_mesh(gen, seed, z_, l, j, w, intri, inv_intri, meshes)

                nerf_imgs.append(out)
        else:
            raise ValueError("invalid mode")
        nerf_imgs = np.array(nerf_imgs)
        save_path = f"{config.out_root}/result/{config.out}/{mode}_video/"
        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/{img_id}.npy", nerf_imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/NARF_GAN/AIST/20220226_aist_triplane_centerFixed.yml")
    parser.add_argument('--default_config', type=str, default="configs/NARF_GAN/default.yml")
    parser.add_argument('--person_id', type=int, action='append')
    parser.add_argument('--iteration', type=str, default="latest")
    parser.add_argument('--psi', type=float, default=0.7)

    args = parser.parse_args()

    mesh_th = 5
    render_size = 256
    whiten_bg = False
    truncation_psi = args.psi

    main(mode="pose_manipulation")
    main(mode="rotation")
    main(mode="appearance_interpolation")
