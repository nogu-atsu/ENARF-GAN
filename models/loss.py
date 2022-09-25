import torch
import torch.nn.functional as F


def push_to_background(fake_mask, background_ratio=0.3):
    if background_ratio > 0:
        fake_mask = fake_mask.reshape(-1)
        fake_mask = torch.topk(fake_mask, k=int(torch.numel(fake_mask) * background_ratio), largest=False,
                               sorted=False)[0]
        loss = (fake_mask ** 2).mean()
    else:
        loss = 0
    return loss


def nerf_bone_loss(fake_mask, bone_mask):
    assert fake_mask.ndim == bone_mask.ndim
    if fake_mask.shape[-1] != bone_mask.shape[-1]:
        downscale_rate = bone_mask.shape[-1] // fake_mask.shape[-1]
        bone_mask = F.max_pool2d(bone_mask[:, None], downscale_rate, downscale_rate, 0)[:, 0]

    binary_bone_mask = bone_mask > 0.5
    loss = ((1 - fake_mask) ** 2 * binary_bone_mask).sum() / binary_bone_mask.sum()
    return loss


def nerf_patch_loss(fake_mask, bone_mask, background_ratio=0.3, coef=10):
    loss = push_to_background(fake_mask, background_ratio=background_ratio) + \
           nerf_bone_loss(fake_mask, bone_mask)
    return loss * coef
