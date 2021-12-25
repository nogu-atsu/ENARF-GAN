import math

import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from NARF.models.net import NeRF
from NARF.models.model_utils import whole_image_grid_ray_sampler
from models.stylegan import Generator as StyleGANGenerator
from models.stylegan import StyledConv, ModulatedConv2d
from utils.rotation_utils import rotation_6d_to_matrix


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


class NeRFNRGenerator(nn.Module):  # NeRF + Neural Rendering
    def __init__(self, config, size, num_bone=1, parent_id=None, num_bone_param=None):
        super(NeRFNRGenerator, self).__init__()
        self.config = config
        self.size = size
        self.num_bone = num_bone
        self.ray_sampler = whole_image_grid_ray_sampler

        z_dim = config.z_dim
        hidden_size = config.nerf_params.hidden_size
        nerf_out_dim = config.nerf_params.out_dim
        patch_size = config.patch_size

        self.nerf = NeRF(config.nerf_params, z_dim=z_dim, num_bone=num_bone, bone_length=True,
                         parent=parent_id, num_bone_param=num_bone_param)
        self.background_generator = StyleGANGenerator(size=patch_size, style_dim=z_dim,
                                                      n_mlp=4, last_channel=nerf_out_dim)

        self.neural_renderer = NeuralRenderer(nerf_out_dim, hidden_size,
                                              num_upsample=int(math.log2(self.size // patch_size)))

    def normalized_inv_intrinsics(self, intrinsics: torch.tensor):
        normalized_intrinsics = torch.cat([intrinsics[:2] / self.size, intrinsics.new([[0, 0, 1]])], dim=0)
        normalized_inv_intri = torch.linalg.inv(normalized_intrinsics)
        return normalized_inv_intri

    @property
    def memory_cost(self):
        return self.nerf.memory_cost

    @property
    def flops(self):
        return self.nerf.flops

    def forward(self, pose_to_camera, pose_to_world, bone_length, z=None, inv_intrinsics=None,
                return_intermediate=False):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param pose_to_world: wold coordinate of joint
        :param bone_length:
        :param background:
        :param z: latent vector
        :param inv_intrinsics:
        :param return_intermediate:
        :return:
        """
        assert self.num_bone == 1 or (bone_length is not None and pose_to_camera is not None)
        batchsize = pose_to_camera.shape[0]
        patch_size = self.config.patch_size

        grid, homo_img = self.ray_sampler(self.size, patch_size, batchsize)

        z_dim = z.shape[1] // 4
        z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)

        # sparse rendering
        inv_intrinsics = torch.tensor(inv_intrinsics).float().cuda(homo_img.device)
        nerf_output = self.nerf(batchsize, patch_size ** 2, homo_img,
                                pose_to_camera, inv_intrinsics, z_for_nerf,
                                pose_to_world, bone_length, thres=0.0,
                                Nc=self.config.nerf_params.Nc,
                                Nf=self.config.nerf_params.Nf,
                                return_intermediate=return_intermediate)

        low_res_feature, low_res_mask = nerf_output[:2]
        low_res_feature = low_res_feature.reshape(batchsize, self.nerf.out_dim, patch_size, patch_size)
        low_res_mask = low_res_mask.reshape(batchsize, patch_size, patch_size)

        bg_feature, _ = self.background_generator([z_for_background])

        rendered_color = self.neural_renderer(low_res_feature, low_res_mask,
                                              bg_feature, z_for_neural_render)

        if return_intermediate:
            fine_points, fine_density = nerf_output[-1]
            return rendered_color, low_res_mask, fine_points, fine_density

        return rendered_color, low_res_mask


class Encoder(nn.Module):
    def __init__(self, config, parents: np.ndarray):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        # remove MaxPool, AvgPool, and linear
        self.image_encoder = nn.Sequential(*[l for l in resnet.children() if not isinstance(l, nn.MaxPool2d)][:-2])
        self.image_feat_dim = config.image_feat_dim
        self.image_size = config.image_size
        self.z_dim = config.z_dim
        self.num_bone = parents.shape[0]

        transformer_hidden_dim = config.transformer_hidden_dim
        transformer_n_head = config.transformer_n_head
        num_encoder_layers = config.num_encoder_layers
        num_decoder_layers = config.num_decoder_layers

        self.transformer = nn.Transformer(d_model=transformer_hidden_dim, nhead=transformer_n_head,
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=transformer_hidden_dim)

        self.image_positional_encoding = nn.Parameter(torch.randn((self.image_size // 16) ** 2, 1,
                                                                  transformer_hidden_dim))
        self.pose_positional_encoding = nn.Parameter(torch.randn(self.num_bone, 1,
                                                                 transformer_hidden_dim))

        self.linear_image = nn.Linear(self.image_feat_dim, transformer_hidden_dim)
        self.linear_pose = nn.Linear(2, transformer_hidden_dim)
        self.linear_out = nn.Linear(transformer_hidden_dim, 7)

        # z
        self.conv_z = nn.Conv2d(self.image_feat_dim, self.z_dim * 4, 3, 2, 1)
        self.linear_z = nn.Linear(self.z_dim * 4, self.z_dim * 4)

        # self.intrinsic = intrinsic.astype("float32")
        self.parents = parents  # parent ids
        self.mean_bone_length = 0.15  # mean length of bone. Should be calculated from dataset??

    def scale_pose(self, pose):
        bone_length = torch.linalg.norm(pose[:, 1:] - pose[:, self.parents[1:]], dim=2)[:, :, 0]  # (B, n_parts-1)
        mean_bone_length = bone_length.mean(dim=1)
        return pose / mean_bone_length[:, None, None, None] * self.mean_bone_length

    def get_rotation_matrix(self, d6: torch.Tensor, pose_translation: torch.tensor,
                            parent: torch.tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)
            pose_translation: (B, n_parts, 3, 1)
            parent:
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        parent_child = pose_translation[:, self.parents[1:], :, 0] - pose_translation[:, 1:, :, 0]
        a1 = torch.cat([a1[:, :1], parent_child], dim=1)  # (B, num_parts, 3)
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, img: torch.tensor, pose_2d: torch.tensor, intrinsic: torch.tensor):
        """estimate 3d pose from image and GT 2d pose

        Args:
            img: (B, 3, 128, 128)
            pose_2d: (B, n_parts, 2)

        Returns:

        """
        intrinsic = torch.tensor(intrinsic, device=img.device)
        inv_intrinsic = torch.inverse(intrinsic)[:, None]
        batchsize = img.shape[0]
        feature_size = self.image_size // 16

        # resnet
        img = (img * 0.5 + 0.5 - self.resnet_mean) / self.resnet_std
        image_feature = self.image_encoder(img)  # (B, 512)

        # z
        h = self.conv_z(image_feature)  # (B, ?, 8, 8)
        z = self.linear_z(h.mean(dim=(2, 3)))  # (B, z_dim * 4)

        # pose
        image_feature = image_feature.reshape(batchsize, self.image_feat_dim, feature_size ** 2)
        image_feature = image_feature.permute(2, 0, 1)  # (64, B, 512)
        image_feature = self.linear_image(image_feature)

        normalized_pose_2d = pose_2d / (self.image_size / 2) - 1

        pose_feature = normalized_pose_2d.permute(1, 0, 2)  # (n_parts, B, 2)
        pose_feature = self.linear_pose(pose_feature)

        # positional encoding
        image_feature = image_feature + self.image_positional_encoding
        pose_feature = pose_feature + self.pose_positional_encoding

        h = self.transformer(image_feature, pose_feature)  # (n_parts, B, transformer_hidden_dim)
        h = self.linear_out(h)  # (n_parts, B, 7)

        rot = h[..., :6]  # (n_parts, B, 6)
        depth = F.softplus(h[..., 6])  # (n_parts, B)

        rotation_matrix = rotation_6d_to_matrix(rot)  # (n_parts, B, 3, 3)
        rotation_matrix = rotation_matrix.permute(1, 0, 2, 3)

        pose_homo = torch.cat([pose_2d, torch.ones_like(pose_2d[:, :, :1])], dim=2) * depth.permute(1, 0)[:, :, None]
        pose_translation = torch.matmul(inv_intrinsic,
                                        pose_homo[:, :, :, None])  # (B, n_parts, 3, 1)
        pose_translation = self.scale_pose(pose_translation)

        pose_Rt = torch.cat([rotation_matrix, pose_translation], dim=-1)  # (B, n_parts, 3, 4)
        num_parts = pose_Rt.shape[1]
        pose_Rt = torch.cat([pose_Rt,
                             torch.tensor([0, 0, 0, 1],
                                          device=pose_Rt.device)[None, None, None].expand(batchsize, num_parts, 1, 4)],
                            dim=2)

        # bone length
        coordinate = pose_Rt[:, :, :3, 3]
        length = torch.linalg.norm(coordinate[:, 1:] - coordinate[:, self.parents[1:]], axis=2)
        bone_length = length[:, :, None]
        return pose_Rt, z, bone_length, intrinsic


class PoseDiscriminator(nn.Module):
    def __init__(self, num_bone: int, n_mlp: int = 4, hidden_dim: int = 256):
        super(PoseDiscriminator, self).__init__()
        layers = [nn.Linear(2 * num_bone, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(n_mlp - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, pose_2d: torch.tensor):
        """pose discriminator

        Args:
            pose_2d: (b, n_parts, 2), normalized to [-1, 1]

        Returns:

        """
        batchsize = pose_2d.shape[0]
        pose_2d = pose_2d.reshape(batchsize, -1)
        out = self.model(pose_2d)
        return out
