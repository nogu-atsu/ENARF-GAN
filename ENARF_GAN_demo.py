import argparse
import os
import pickle
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from libraries.config import yaml_config
from models.generator import TriNARFGenerator

warnings.filterwarnings('ignore')


def main():
    config_path = args.config
    default_config_path = "configs/enarfgan/default.yml"
    config = yaml_config(config_path, default_config_path)

    size = config.dataset.image_size

    gen_config = config.generator_params

    num_bone = 24
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13,
                        14, 16, 17, 18, 19, 20, 21])

    canonical_pose_path = f"{config.sample_path}/canonical.npy"
    canonical_pose = np.load(canonical_pose_path)

    gen = TriNARFGenerator(gen_config, size, num_bone=num_bone,
                           num_bone_param=num_bone - 1,
                           parent_id=parents,
                           black_background=False)
    gen.register_canonical_pose(canonical_pose)
    gen.to("cuda")

    snapshot_path = f"{config.out_root}/result/{config.out}/snapshot_latest.pth"
    if os.path.exists(snapshot_path):
        state_dict = torch.load(snapshot_path)["gen"]
        gen.load_state_dict(state_dict, strict=False)
    else:
        raise Exception("model not loaded")

    gen.eval()
    mesh = True

    with open(f"{config.sample_path}/sample_data.pickle", "rb") as f:
        samples = pickle.load(f)

    save_dir = f"{config.out_root}/result/{config.out}/samples"
    os.makedirs(save_dir, exist_ok=True)
    for idx, data in tqdm(enumerate(samples)):
        data = {key: torch.tensor(val).cuda(non_blocking=True).float() for key, val in data.items()}
        pose_to_camera = data["pose_to_camera"][None]
        bone_length = data["bone_length"][None]
        intrinsic = data["intrinsics"][None]
        inv_intrinsic = torch.inverse(intrinsic)

        z_dim = gen.config.z_dim * 4
        z = torch.cuda.FloatTensor(1, z_dim).normal_()
        with torch.no_grad():
            fake_fg, fake_mask, fake_bg = gen(pose_to_camera, None, bone_length, z, inv_intrinsic, truncation_psi=0.4,
                                              return_bg=True, )
            if mesh:
                mesh_img, meshes = gen.render_mesh(pose_to_camera, intrinsic, z, bone_length, mesh_th=5,
                                                   truncation_psi=0.4)
            if fake_fg.shape[-1] == fake_mask.shape[-1]:
                fake_img = fake_fg + (1 - fake_mask) * fake_bg
            else:
                fake_img = fake_fg

            fake_img = fake_img.cpu().numpy()[0].transpose(1, 2, 0)
            fake_img = fake_img * 127.5 + 127.5
            fake_img = np.clip(fake_img, 0, 255).astype("uint8")

            fake_mask = (fake_mask.cpu().numpy()[0] * 255).astype("uint8")

        Image.fromarray(fake_img).save(f"{save_dir}/img_{idx:0>4}.png")
        Image.fromarray(fake_mask).save(f"{save_dir}/mask_{idx:0>4}.png")
        Image.fromarray(mesh_img).save(f"{save_dir}/mesh_{idx:0>4}.png")
    print(f"Images are saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    main()
