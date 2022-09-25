import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import lpips as _lpips
from lpips_pytorch import LPIPS


def ssim(img1, img2):
    img1 = img1[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # size x size x 3
    img2 = img2[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # size x size x 3
    return structural_similarity(img1, img2, data_range=1, multichannel=True)  # scalar


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    return 20 * np.log10(2) - 10 * np.log10(mse)


loss_fn_vgg = _lpips.LPIPS(net='vgg').cuda()


def lpips(img1, img2):
    lp = loss_fn_vgg(img1, img2)
    return lp


neural_actor_lpips_func = LPIPS(net_type='alex', version='0.1').cuda()


def neural_actor_lpips(img1, img2):
    lp = neural_actor_lpips_func(img1, img2)
    return lp
