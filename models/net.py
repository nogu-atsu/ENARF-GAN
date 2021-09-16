import numpy as np
import torch
from torch import nn

from NARF.models.net import get_nerf_module
from NARF.models.tiny_utils import whole_image_grid_ray_sampler
from models.stylegan import Generator as StyleGANGenerator
from models.stylegan import StyledConv, ModulatedConv2d


class NeuralRenderer(nn.Module):
    def __init__(self, in_channel, style_dim, channel_multiplier: int = 32, input_size: int = 32,
                 num_upsample: int = 2, blur_kernel: list = [1, 3, 3, 1]):
        super(NeuralRenderer, self).__init__()

        size = input_size

        _channel = channel_multiplier * 2 ** num_upsample

        layers = [StyledConv(in_channel, _channel, kernel_size=3, style_dim=style_dim,
                             upsample=False, demodulate=True, use_noise=False)]

        for i in range(num_upsample):
            layers += [
                StyledConv(_channel, _channel, kernel_size=3, style_dim=style_dim,
                           upsample=True, blur_kernel=blur_kernel, demodulate=True, use_noise=False),
                StyledConv(_channel, _channel // 2, kernel_size=3, style_dim=style_dim,
                           upsample=False, demodulate=True, use_noise=False)
            ]
            size = size * 2
            _channel = _channel // 2
        self.layers = nn.ModuleList(layers)
        self.conv = ModulatedConv2d(_channel, 4, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, fg_feat: torch.tensor, mask: torch.tensor, bg_feat: torch.tensor, z: torch.tensor
                ) -> torch.tensor:
        h = fg_feat + (1 - mask[:, None]) * bg_feat
        for l in self.layers:
            h = l(h, z)
        h = self.conv(h, z) + self.bias
        color = h[:, :3]
        return color


class NeRFNRGenerator(nn.Module):
    def __init__(self, config, size, intrinsics=None, num_bone=1, parent_id=None):
        super(NeRFNRGenerator, self).__init__()
        self.config = config
        self.size = size
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        normalized_intrinsics = np.concatenate([intrinsics[:2] / size, np.array([[0, 0, 1]])], axis=0)
        self.normalized_inv_intrinsics = np.linalg.inv(normalized_intrinsics)
        self.num_bone = num_bone
        self.ray_sampler = whole_image_grid_ray_sampler

        z_dim = config.z_dim
        hidden_size = config.nerf_params.hidden_size
        nerf_out_dim = config.nerf_params.out_dim

        nerf = get_nerf_module(config)
        self.nerf = nerf(config.nerf_params, z_dim=z_dim, groups=num_bone, bone_length=True,
                         parent=parent_id)
        self.background_generator = StyleGANGenerator(size=size // 4, style_dim=z_dim,
                                                      n_mlp=4, last_channel=nerf_out_dim)

        self.neural_renderer = NeuralRenderer(nerf_out_dim, hidden_size)

    @property
    def memory_cost(self):
        return self.nerf.memory_cost

    @property
    def flops(self):
        return self.nerf.flops

    def forward(self, pose_to_camera, pose_to_world, bone_length, z=None, inv_intrinsics=None):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param pose_to_world: wold coordinate of joint
        :param bone_length:
        :param background:
        :param z: latent vector
        :param inv_intrinsics:
        :return:
        """
        assert self.num_bone == 1 or (bone_length is not None and pose_to_camera is not None)
        batchsize = pose_to_camera.shape[0]
        patch_size = self.config.patch_size

        grid, homo_img = self.ray_sampler(self.size, patch_size, batchsize)

        z_dim = z.shape[1] // 4
        z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)

        # sparse rendering
        if inv_intrinsics is None:
            inv_intrinsics = self.inv_intrinsics
        inv_intrinsics = torch.tensor(inv_intrinsics).float().cuda(homo_img.device)
        low_res_feature, low_res_mask = self.nerf(batchsize, patch_size ** 2, homo_img,
                                                  pose_to_camera, inv_intrinsics, z_for_nerf,
                                                  pose_to_world, bone_length, thres=0.0,
                                                  Nc=self.config.nerf_params.Nc,
                                                  Nf=self.config.nerf_params.Nf)
        low_res_feature = low_res_feature.reshape(batchsize, self.nerf.out_dim, patch_size, patch_size)
        low_res_mask = low_res_mask.reshape(batchsize, patch_size, patch_size)

        bg_feature, _ = self.background_generator([z_for_background])

        rendered_color = self.neural_renderer(low_res_feature, low_res_mask,
                                              bg_feature, z_for_neural_render)

        return rendered_color, low_res_mask
