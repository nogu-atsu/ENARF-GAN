import os, argparse
import pickle
import warnings

import numpy as np
import torch
from PIL import Image

from libraries.config import yaml_config
from models.generator import SSONARFGenerator

warnings.filterwarnings('ignore')


def main():
    config_path = args.config
    default_config_path = "configs/DSO/default.yml"
    config = yaml_config(config_path, default_config_path)

    size = config.dataset.image_size

    gen_config = config.generator_params
    gen_config.nerf_params.render_bs = 16384

    num_bone = 24
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13,
                        14, 16, 17, 18, 19, 20, 21])

    canonical_pose_path = f"smpl_data/neutral_canonical.npy"
    canonical_pose = np.load(canonical_pose_path)
    gen = SSONARFGenerator(config.generator_params, size, num_bone,
                           parent_id=parents, num_bone_param=num_bone - 1)
    gen.register_canonical_pose(canonical_pose)

    gen.to("cuda").eval()

    snapshot_path = f"{config.out_root}/result/{config.out}/snapshot_latest.pth"
    if os.path.exists(snapshot_path):
        state_dict = torch.load(snapshot_path)["gen"]
        gen.load_state_dict(state_dict, strict=False)
    else:
        raise Exception("model not loaded")

    bg_color = config.dataset.bg_color

    frame_time = torch.tensor([1.]).cuda(non_blocking=True)

    with open(f"{config.sample_path}/sample_data.pickle", "rb") as f:
        samples = pickle.load(f)

    save_dir = f"{config.out_root}/result/{config.out}/samples"
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(len(samples)):
        batch = samples[idx]
        batch = {key: torch.tensor(val).cuda(non_blocking=True).float() for key, val in batch.items()}

        pose_to_camera = batch["pose_3d"][None]
        bone_length = batch["bone_length"][None]
        intrinsic = batch["intrinsics"][None]
        inv_intrinsic = torch.inverse(intrinsic)

        nerf_color, nerf_mask, grid = gen.render_entire_img(pose_to_camera, inv_intrinsic, frame_time,
                                                            bone_length, None, size, no_grad=True)

        nerf_color = nerf_color + bg_color * (1 - nerf_mask)
        nerf_color = nerf_color.cpu().numpy().transpose(1, 2, 0)
        nerf_color = nerf_color * 127.5 + 127.5
        nerf_color = np.clip(nerf_color, 0, 255).astype("uint8")

        Image.fromarray(nerf_color).save(f"{save_dir}/{idx:0>4}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    main()
