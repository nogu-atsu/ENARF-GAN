import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(".")
from cuda_extension.triplane_sampler import grid_sampler, triplane_sampler


# position_2d = position[:, [0, 1, 1, 2, 2, 0]].reshape(batchsize * 3, 2, n)
# position_2d = position_2d.permute(0, 2, 1)[:, :, None]  # (B * 3, n, 1, 2)
#
# # if batch_idx is not None, place tri-planes side by side to form a single tri-plane (quite tricky)
# if batch_idx is not None:  # transform x coordinate
#     actual_batchsize = w // (h + 1)
#     scale = 1 / (actual_batchsize * (1 + 1 / h))
#     position_2d[:, :, :, 0] = (position_2d[:, :, :, 0] * scale +
#                                batch_idx[None, :, None] * (2 / actual_batchsize) + (scale - 1))
#
# feature = F.grid_sample(features, position_2d, align_corners=False)
# # feature = torch.cudnn_grid_sampler(features, position_2d)
# feature = feature.reshape(batchsize, 3, -1, n)
# if reduction == "sum":
#     feature = feature.sum(dim=1)  # (B, feat_dim, n)
# elif reduction == "prod":
#     if self.config.clamp_mask:
#         feature = (feature.data.clamp(-2, 5) - feature.data) + feature
#     feature = torch.sigmoid(feature).prod(dim=1)
# else:
#     raise ValueError()
# return feature

def test_triplane_sampler():
    bs = 1
    ch = 32
    in_h, in_w = 256, 256
    out_h, out_w = 3200, 10000
    a = torch.empty((bs, ch * 3, in_h, in_w), dtype=torch.float32, device="cuda").uniform_().requires_grad_(False)
    index = torch.empty((bs, out_h, out_w, 3), dtype=torch.float32, device="cuda").uniform_(-1, 1).requires_grad_(False)

    n_loop = 10

    # pytorch version
    torch.cuda.synchronize()
    s = time.time()
    for i in range(n_loop):
        torch.cuda.synchronize()
        index_tri = index[:, :, :, [0, 1, 1, 2, 2, 0]
                    ].reshape(bs, out_h, out_w, 3, 2).permute(0, 3, 1, 2, 4).reshape(bs * 3, out_h, out_w, 2)
        a_tri = a.reshape(bs * 3, ch, in_h, in_w)
        out1 = F.grid_sample(a_tri, index_tri, mode="bilinear", align_corners=False)
        out1 = out1.reshape(bs, 3, ch, out_h, out_w).sum(dim=1)
        if out1.requires_grad:
            out1.sum().backward()
    torch.cuda.synchronize()
    print("case 1:", time.time() - s)
    if a.requires_grad:
        pytorch_grad_a = a.grad.clone()
    if index.requires_grad:
        pytorch_grad_index = index.grad.clone()
    pytorch_out = out1.clone()

    a.grad = None
    index.grad = None

    # cpp version
    torch.cuda.synchronize()
    s = time.time()
    for i in range(n_loop):
        torch.cuda.synchronize()
        out1 = triplane_sampler(a, index, mode="bilinear", align_corners=False)
        if out1.requires_grad:
            out1.sum().backward()
    torch.cuda.synchronize()
    print("cpp:", time.time() - s)
    cpp_grad_a = a.grad
    cpp_grad_index = index.grad
    if cpp_grad_a is not None:
        print(torch.isclose(pytorch_grad_a, cpp_grad_a).all())
    if cpp_grad_index is not None:
        print(torch.isclose(pytorch_grad_index, cpp_grad_index).all())
    print(torch.isclose(pytorch_out, out1).all())


def test_grid_sampler():
    a = torch.empty((3, 32, 256, 256), dtype=torch.float32, device="cuda").uniform_().requires_grad_(True)
    index = torch.empty((3, 320, 1000, 2), dtype=torch.float32, device="cuda").uniform_(-1, 1).requires_grad_(False)

    n_loop = 100

    # pytorch version
    torch.cuda.synchronize()
    s = time.time()
    for i in range(n_loop):
        torch.cuda.synchronize()
        out1 = F.grid_sample(a, index, mode="bilinear", align_corners=False)
        out1.sum().backward()
    torch.cuda.synchronize()
    print("case 1:", time.time() - s)
    pytorch_grad_a = a.grad.clone()
    # pytorch_grad_index = index.grad.clone()

    a.grad = None
    index.grad = None

    # cpp version
    Bilinear = 0
    Zeros = 0
    torch.cuda.synchronize()
    s = time.time()
    for i in range(n_loop):
        torch.cuda.synchronize()
        out1 = grid_sampler(a, index, mode="bilinear", align_corners=False)
        out1.sum().backward()
    torch.cuda.synchronize()
    print("cpp:", time.time() - s)
    cpp_grad_a = a.grad
    cpp_grad_index = index.grad

    # print(torch.isclose(pytorch_grad_a, cpp_grad_a).all())
    # print(torch.isclose(pytorch_grad_index, cpp_grad_index).all())

    # print(cpp_grad_a)
    print(cpp_grad_index)


if __name__ == "__main__":
    test_triplane_sampler()
