import glob
import pickle
import sys

import numpy as np
import torch
from smplx.body_models import SMPL
from tqdm import tqdm

sys.path.append("../../")
from libraries.smpl_utils import get_pose

smpl_data_path = "/data/unagi0/noguchi/dataset/mosh/neutrMosh/neutrSMPL_CMU/**/*.pkl"
smpl_data_path = glob.glob(smpl_data_path, recursive=True)

smpl = SMPL(model_path="../../smpl_data")

trans_list = []
poses_list = []
betas_list = []

for path in tqdm(smpl_data_path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding='latin1')
    video_len = data['poses'].shape[0]
    trans_list.append(data['trans'])
    poses_list.append(data['poses'])
    betas_list.append(data['betas'][None].repeat(video_len, axis=0))

thin_out_rate = 10
trans = np.concatenate(trans_list, axis=0)[::thin_out_rate]
poses = np.concatenate(poses_list, axis=0)[::thin_out_rate].reshape(-1, 24, 3)
betas = np.concatenate(betas_list, axis=0)[::thin_out_rate]

A = get_pose(smpl, body_pose=torch.tensor(poses[:, 1:, :]).float(),
             betas=torch.tensor(betas).float(),
             global_orient=torch.tensor(poses[:, 0:1, :]).float() * 0)
A[:, :, :3] *= torch.tensor([1, -1, -1])[None, None, :, None]
A = A.numpy()
np.save("/data/unagi0/noguchi/dataset/mosh/CMU/bone_params_128.npy", A)
