import time
import warnings

import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

batch_size = 12
features = torch.randn(batch_size * 23 * 3, 1, 256, 256, device="cuda").requires_grad_(True)
rand = torch.ones(batch_size * 23 * 3, 1, 128 ** 2 * 64, 2, device="cuda").requires_grad_(True) * 3


def grid_sample(features, val):
    torch.cuda.synchronize()
    s = time.time()
    feature = F.grid_sample(features, val, align_corners=False)

    torch.cuda.synchronize()
    print("forward", time.time() - s)
    s = time.time()
    loss = feature.sum()
    loss.backward()
    torch.cuda.synchronize()
    print("back", time.time() - s)


for val in [rand]:
    grid_sample(features, val)
