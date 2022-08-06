import sys
from smplx.body_models import SMPL
import numpy as np
import torch

sys.path.append("../")
from libraries.smpl_utils import get_pose

for gender in ["MALE", "FEMALE", "NEUTRAL"]:
    smpl = SMPL(model_path="../../smpl_data", gender='MALE', batch_size=1)
    poses = np.zeros((1, 24, 3))
    with torch.no_grad():
        A = get_pose(smpl, body_pose=torch.tensor(poses[:, 1:]).float(),
                     global_orient=torch.tensor(poses[:, 0:1, :]).float())
        A[:, :, :3, 3] = A[:, :, :3, 3] - A[:, [1, 2], :3, 3].mean(dim=1, keepdim=True)
        A = A.numpy()[0]

    np.save(f'../smpl_data/{gender.lower()}_canonical.npy', A)
