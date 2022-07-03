import os

import warnings

warnings.filterwarnings('ignore')
import sys

sys.path.append(".")
import time
from dataset.dataset import SSODataset
from models.generator import SSONARFGenerator
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dependencies.config import yaml_config

render_size = 128
batchsize = 1
# config_path = "configs/NARF_GAN/THUman/20210913_NoPoseAndRayDirection.yml" # using old dataset (19 parts?)
# config_path = "configs/NARF_GAN/THUman/20211213_CMUPrior.yml"
# config_path = "configs/NARF_GAN/ZJU/20211221_zju.yml"
# config_path = "configs/NARF_GAN/ZJU/20211223_zju.yml"
# config_path = "configs/NARF_GAN/ZJU/20211226_zju_aligned_l2.yml"

config_path = "configs/SSO/ZJU/20220215_zju313_triplane.yml"

# config_path = "configs/SSO/ZJU/20220215_zju313_triplane_const_tri.yml"
# config_path = "configs/SSO/ZJU/20220215_zju315_const_triplane.yml"
# config_path = "configs/SSO/ZJU/20220215_zju315_const_trimask.yml"
# config_path = "configs/SSO/ZJU/20220215_zju315_deform_triplane.yml"
# config_path = "configs/SSO/ZJU/20220221_zju315_deform_triplane.yml"
# config_path = "configs/SSO/ZJU/20220220_zju315_narf.yml"
# config_path = "configs/SSO/ZJU/20220220_zju315_tnarf.yml"
# config_path = "configs/SSO/ZJU/20220220_zju377_narf.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220226_zju386_tpenarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju386_enarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju386_narf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220226_zju315_tpenarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju315_enarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju315_narf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju315_senarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220226_zju315_enarf_wo_selector_centerFixed.yml"

# config_path = "configs/SSO/ZJU/cvpr_exp/20220226_zju313_tpenarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju313_enarf_centerFixed.yml"
config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju313_narf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju313_senarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220226_zju313_enarf_wo_selector_centerFixed.yml"

# config_path = "configs/SSO/ZJU/cvpr_exp/20220226_zju386_tpenarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju386_enarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju386_narf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju386_senarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220226_zju386_enarf_wo_selector_centerFixed.yml"

# config_path = "configs/SSO/ZJU/cvpr_exp/20220225_zju313_enarf_centerFixed.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220228_zju386_enarf_centerFixed_surfReg.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220301_zju315_enarf_centerFixed_scale1.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220301_zju386_enarf_centerFixed_scale1.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220303_zju313_enarf_centerFixed_noRay.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220303_zju313_tpenarf_centerFixed_noRay.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220303_zju313_senarf_centerFixed_noRay.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220303_zju315_enarf_centerFixed_noRay.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220303_zju315_tpenarf_centerFixed_noRay.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220303_zju386_enarf_centerFixed_noRay.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220303_zju386_tpenarf_centerFixed_noRay.yml"
# config_path = "configs/SSO/ZJU/cvpr_exp/20220303_zju386_senarf_centerFixed_noRay.yml"


default_config_path = "configs/SSO/default.yml"

config = yaml_config(config_path, default_config_path)

size = config.dataset.image_size
train_dataset_config = config.dataset.train
train_dataset_config = config.dataset.val.novel_pose
# train_dataset_config = config.dataset.val.novel_view

# fix n_camerapqb
if "386" in config_path:
    train_dataset_config.n_camera = 19

dataset = SSODataset(train_dataset_config, size=size, return_bone_params=True,
                     return_bone_mask=True, random_background=False, just_cache=False,
                     load_camera_intrinsics=True)
# PoseDataset = HumanPoseDataset if config.dataset.name == "human_v2" else THUmanPoseDataset

# dataset = PoseDataset(size=size, data_root=data_root)
loader = DataLoader(dataset, batch_size=batchsize, num_workers=1, shuffle=True, drop_last=True)

gen_config = config.generator_params
gen_config.nerf_params.render_bs = 16384
# gen_config.nerf_params.coordinate_scale = 0.3


num_bone = dataset.num_bone
gen = SSONARFGenerator(config.generator_params, size, num_bone,
                       parent_id=dataset.parents, num_bone_param=dataset.num_bone_param)
gen.register_canonical_pose(dataset.canonical_pose)

gen.to("cuda")

# gen = NeRFNRGenerator(config.generator_params, size,
# num_bone=dataset.num_bone, num_bone_param=dataset.num_bone_param, parent_id=dataset.parents).to("cuda")
if os.path.exists(f"{config.out_root}/result/{config.out}/snapshot_latest.pth"):
    state_dict = torch.load(f"{config.out_root}/result/{config.out}/snapshot_latest.pth")["gen"]
    # for k in list(state_dict.keys()):
    #     if "activate.bias" in k:
    #         state_dict[k[:-13]+"bias"] = state_dict[k].reshape(1, -1, 1, 1)
    #         del state_dict[k]
    gen.load_state_dict(state_dict, strict=False)
else:
    print("model not loaded")

gen.eval()
num_bone = dataset.num_bone

num = 1
start = 2
cam_id = 8
plt.figure(figsize=(32, 16))

rendering_time = []

batch = loader.dataset[(start) * train_dataset_config.n_camera + cam_id]
batch = {key: torch.tensor(val).cuda(non_blocking=True).float() for key, val in batch.items()}

img = batch["img"]
mask = batch["mask"][None]
pose_to_camera = batch["pose_3d"][None]
frame_time = batch["frame_time"][None]
bone_length = batch["bone_length"][None]
camera_rotation = batch["camera_rotation"][None]
intrinsic = batch["intrinsics"][None]
inv_intrinsic = torch.inverse(intrinsic)
# with torch.no_grad():
torch.cuda.synchronize()
s = time.time()

gen.profile_memory_stats(pose_to_camera, inv_intrinsic, frame_time,
                                                    bone_length, camera_rotation, size)
