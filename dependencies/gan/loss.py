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
