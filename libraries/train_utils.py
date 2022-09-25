import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image


def record_command(out):
    with open(out + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")


def write(iter, loss, name, writer):
    writer.add_scalar("metrics/" + name, loss, iter)
    return loss


def all_reduce(scalar):
    scalar = torch.tensor(scalar).cuda()
    torch.distributed.all_reduce(scalar)
    return scalar.item()


def save_img(batch, name):  # b x 3 x size x size
    if isinstance(batch, torch.Tensor):
        batch = batch.data.cpu().numpy()
    if len(batch.shape) == 3:
        batch = np.tile(batch[:, None], (1, 3, 1, 1))
    b, _, size, _ = batch.shape
    n = int(b ** 0.5)

    batch = batch.transpose(0, 2, 3, 1)
    batch = batch[:n ** 2].reshape(n, n, size, size, 3)
    batch = np.concatenate(batch, axis=1)
    batch = np.concatenate(batch, axis=1)
    batch = np.clip(batch * 127.5 + 127.5, 0, 255).astype("uint8")
    batch = Image.fromarray(batch)
    batch.save(name)
