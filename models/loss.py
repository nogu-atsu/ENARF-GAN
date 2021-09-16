import torch
import torch.nn.functional as F
from torch import autograd


def adv_loss_dis(real, fake, adv_loss_type, tmp=1.0):
    if adv_loss_type == "hinge":
        return F.relu(1 - real).mean() + F.relu(1 + fake).mean()
    elif adv_loss_type == "ce":
        return F.softplus(-real * tmp).mean() + F.softplus(fake * tmp).mean()
    else:
        assert False, f"{adv_loss_type} is not supported"


def adv_loss_gen(fake, adv_loss_type, tmp=1.0):
    if adv_loss_type == "hinge":
        return -fake.mean()
    elif adv_loss_type == "ce":
        return F.softplus(-fake * tmp).mean()
    else:
        assert False, f"{adv_loss_type} is not supported"


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def push_to_background(fake_mask, background_ratio=0.3):
    fake_mask = fake_mask.reshape(-1)
    fake_mask = torch.topk(fake_mask, k=int(torch.numel(fake_mask) * background_ratio), largest=False,
                           sorted=False)[0]
    loss = (fake_mask ** 2).mean()
    return loss


def nerf_bone_loss(fake_mask, bone_mask):
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
