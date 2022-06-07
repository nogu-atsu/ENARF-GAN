import math
import time
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from NARF.models.model_utils import whole_image_grid_ray_sampler
from dependencies.NARF.ray_sampler import mask_based_sampler
from NARF.models.net import NeRF
from models.stylegan import Generator as StyleGANGenerator
from models.stylegan import StyledConv, ModulatedConv2d, Blur
from models.nerf_model import StyleNeRF, TriPlaneNARF, SSONARF
from utils.rotation_utils import rotation_6d_to_matrix


class NeuralRenderer(nn.Module):
    def __init__(self, in_channel, style_dim, channel_multiplier: int = 32, input_size: int = 32,
                 num_upsample: int = 2, blur_kernel: list = [1, 3, 3, 1], bg_activation=None):
        super(NeuralRenderer, self).__init__()
        assert bg_activation in ["tanh", "l2", None]
        size = input_size
        self.in_channel = in_channel

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
        self.bg_activation = bg_activation

    def forward(self, fg_feat: torch.tensor, mask: torch.tensor, bg_feat: torch.tensor, z: torch.tensor
                ) -> torch.tensor:
        if self.bg_activation == "tanh":
            bg_feat = torch.tanh(bg_feat)
        elif self.bg_activation == "l2":
            bg_feat = F.normalize(bg_feat, dim=1)
        h = fg_feat + (1 - mask[:, None]) * bg_feat

        if self.bg_activation == "l2":
            h = h * self.in_channel ** 0.5
        for l in self.layers:
            h = l(h, z)
        h = self.conv(h, z) + self.bias
        color = h[:, :3]
        return color


class StyleNeRFRendererBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, up=False):
        super(StyleNeRFRendererBlock, self).__init__()
        self.conv = StyledConv(in_channel, out_channel, kernel_size=1, style_dim=style_dim, use_noise=False)
        self.up = up

        if up:
            self.conv1 = StyledConv(out_channel, out_channel, kernel_size=1, style_dim=style_dim, use_noise=False)
            self.conv2 = StyledConv(out_channel, out_channel * 4, kernel_size=1, style_dim=style_dim, use_noise=False)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.register_buffer("blur_kernel", torch.tensor([[1, 2, 1],
                                                              [2, 4, 2],
                                                              [1, 2, 1]], dtype=torch.float32)[None, None] / 16)

    def forward(self, x, z):
        h = self.conv(x, z)

        if self.up:
            repeated = h.repeat((1, 4, 1, 1))
            h = self.conv1(h, z)
            h = self.conv2(h, z)
            h = h + repeated
            h = self.pixel_shuffle(h)
            bs, ch, size, _ = h.shape
            h = F.conv2d(h.view(bs * ch, 1, size, size), self.blur_kernel, padding=1).view(bs, ch, size, size)
        return h


class StyleNeRFRenderer(nn.Module):
    def __init__(self, in_channel, style_dim, channel_multiplier: int = 32, input_size: int = 32,
                 num_upsample: int = 2, bg_activation=None, ):
        super(StyleNeRFRenderer, self).__init__()
        assert bg_activation in ["tanh", "l2", None]
        self.in_channel = in_channel

        _channel = channel_multiplier * 2 ** num_upsample

        layers = [StyledConv(in_channel, _channel, kernel_size=1, style_dim=style_dim,
                             upsample=False, demodulate=True, use_noise=False)]

        for i in range(num_upsample):
            layers += [
                StyleNeRFRendererBlock(_channel, _channel // 2, style_dim=style_dim, up=True),
                StyledConv(_channel // 2, _channel // 2, kernel_size=1, style_dim=style_dim,
                           upsample=False, demodulate=True, use_noise=False),
            ]
            _channel = _channel // 2

        self.to_rgb = ModulatedConv2d(_channel, 3, kernel_size=1, style_dim=style_dim)

        self.layers = nn.ModuleList(layers)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.bg_activation = bg_activation
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, fg_feat: torch.tensor, mask: torch.tensor, bg_feat: torch.tensor, z: torch.tensor
                ) -> torch.tensor:
        if self.bg_activation == "tanh":
            bg_feat = torch.tanh(bg_feat)
        elif self.bg_activation == "l2":
            bg_feat = F.normalize(bg_feat, dim=1)
        h = fg_feat + (1 - mask[:, None]) * bg_feat

        if self.bg_activation == "l2":
            h = h * self.in_channel ** 0.5

        for l in self.layers:
            h = l(h, z)
        color = self.to_rgb(h, z) + self.bias
        return color


class NARFNRGenerator(nn.Module):  # NeRF + Neural Rendering
    def __init__(self, config, size, num_bone=1, parent_id=None, num_bone_param=None):
        super(NARFNRGenerator, self).__init__()
        self.config = config
        self.size = size
        self.num_bone = num_bone
        self.ray_sampler = whole_image_grid_ray_sampler
        self.background_ratio = config.background_ratio

        z_dim = config.z_dim
        hidden_size = config.nerf_params.hidden_size
        nerf_out_dim = config.nerf_params.out_dim
        patch_size = config.patch_size
        bg_activation = config.bg_activation
        use_style_nerf = config.use_style_nerf
        use_style_nerf_renderer = config.use_style_nerf_renderer
        crop_background = config.crop_background

        nerf_model = StyleNeRF if use_style_nerf else NeRF
        self.nerf = nerf_model(config.nerf_params, z_dim=z_dim, num_bone=num_bone, bone_length=True,
                               parent=parent_id, num_bone_param=num_bone_param)
        self.background_generator = StyleGANGenerator(size=patch_size, style_dim=z_dim,
                                                      n_mlp=6, last_channel=nerf_out_dim,
                                                      crop_background=crop_background)

        renderer = StyleNeRFRenderer if use_style_nerf_renderer else NeuralRenderer
        self.neural_renderer = renderer(nerf_out_dim, hidden_size,
                                        num_upsample=int(math.log2(self.size // patch_size)),
                                        bg_activation=bg_activation)

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
                return_intermediate=False, nerf_scale=1, return_disparity=False, return_bg=False, *args, **kwargs):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param pose_to_world: wold coordinate of joint
        :param bone_length:
        :param z: latent vector
        :param inv_intrinsics:
        :param return_intermediate:
        :param nerf_scale:
        :return:
        """
        assert self.num_bone == 1 or (bone_length is not None and pose_to_camera is not None)
        batchsize = pose_to_camera.shape[0]
        patch_size = self.config.patch_size * 4 if return_disparity else self.config.patch_size

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
                                return_intermediate=return_intermediate,
                                render_scale=nerf_scale,
                                return_disparity=return_disparity)
        if return_disparity:
            disparity = nerf_output[2]
            disparity = disparity * self.config.nerf_params.coordinate_scale
            return None, None, disparity

        low_res_feature, low_res_mask = nerf_output[:2]
        fine_weights = self.nerf.buffers_tensors["fine_weights"]
        fine_depth = self.nerf.buffers_tensors["fine_depth"]

        low_res_feature = low_res_feature.reshape(batchsize, self.nerf.out_dim, patch_size, patch_size)
        low_res_mask = low_res_mask.reshape(batchsize, patch_size, patch_size)

        bg_feature, _ = self.background_generator([z_for_background])

        rendered_color = self.neural_renderer(low_res_feature, low_res_mask,
                                              bg_feature, z_for_neural_render)

        if return_intermediate:
            fine_points, fine_density = nerf_output[-1]
            return rendered_color, low_res_mask, fine_points, fine_density
        if return_bg:
            return rendered_color, low_res_mask, -1
        return rendered_color, low_res_mask, fine_weights, fine_depth

    def render_mesh(self, pose_to_camera, intrinsics, z, bone_length, voxel_size=0.003,
                    mesh_th=15, truncation_psi=0.4):
        z_dim = z.shape[1] // 4
        z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)
        return self.nerf.render_mesh(pose_to_camera, intrinsics, z_for_nerf, bone_length,
                                     voxel_size, mesh_th)


class Encoder(nn.Module):
    def __init__(self, config, parents: np.ndarray):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        # remove MaxPool, AvgPool, and linear
        self.image_encoder = nn.Sequential(*[l for l in resnet.children() if not isinstance(l, nn.MaxPool2d)][:-2])
        self.register_buffer('resnet_mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('resnet_std', torch.tensor([0.229, 0.224, 0.225]))

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
        img = (img * 0.5 + 0.5 - self.resnet_mean[:, None, None]) / self.resnet_std[:, None, None]
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


class ProbablisticEncoder(nn.Module):
    def __init__(self, config, parents: np.ndarray):
        super(ProbablisticEncoder, self).__init__()
        self.image_encoder = torchvision.models.resnet18(pretrained=True)
        self.register_buffer('resnet_mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('resnet_std', torch.tensor([0.229, 0.224, 0.225]))
        self.image_feat_dim = config.image_feat_dim
        self.image_size = config.image_size
        self.z_dim = config.z_dim
        self.num_bone = parents.shape[0]
        self.n_pe_freq = config.n_positional_encoding_freq

        transformer_hidden_dim = config.transformer_hidden_dim
        transformer_n_head = config.transformer_n_head
        num_encoder_layers = config.num_encoder_layers

        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_hidden_dim, nhead=transformer_n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.pose_positional_embedding = nn.Parameter(torch.randn(self.num_bone, 1,
                                                                  transformer_hidden_dim))

        self.linear_image = nn.Linear(self.image_feat_dim, transformer_hidden_dim * self.num_bone)
        self.linear_pose = nn.Linear(2, transformer_hidden_dim)
        self.linear_z_pose = nn.Linear(self.z_dim, transformer_hidden_dim * self.num_bone)

        self.linear_out = nn.Linear(transformer_hidden_dim, 7)

        # z
        self.linear_z1 = nn.Linear(self.image_feat_dim, self.image_feat_dim)
        self.linear_z2 = nn.Linear(self.image_feat_dim, self.z_dim * 4)

        # self.intrinsic = intrinsic.astype("float32")
        self.parents = parents  # parent ids
        self.mean_bone_length = 0.15  # mean length of bone. Should be calculated from dataset??

    def positional_encoding(self, pose: torch.tensor) -> torch.tensor:
        """
        Apply positional encoding to pose
        Args:
            pose: (n_parts, B, 2)

        Returns: (n_parts, B, 4L + 2)

        """
        encoded = [pose]
        encoded += [torch.cos(pose * 2 ** i * np.pi) for i in range(self.n_pe_freq)]
        encoded += [torch.sin(pose * 2 ** i * np.pi) for i in range(self.n_pe_freq)]
        encoded = torch.cat(encoded, dim=-1)
        return encoded

    def scale_pose(self, pose):
        bone_length = torch.linalg.norm(pose[:, 1:] - pose[:, self.parents[1:]], dim=2)[:, :, 0]  # (B, n_parts-1)
        mean_bone_length = bone_length.mean(dim=1)
        return pose / mean_bone_length[:, None, None, None] * self.mean_bone_length

    def get_rotation_matrix(self, d6: torch.Tensor, pose_translation: torch.tensor) -> torch.Tensor:
        """
        Rotation of root joint -> x_axis = d6[:3]
        Rotation of other joints -> x_axis = child_trans - parent_trans

        Args:
            d6: 6D rotation representation, of size (n_parts, B, 6)
            pose_translation: (B, n_parts, 3, 1)
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        """
        d6 = d6.permute(1, 0, 2)
        a1, a2 = d6[..., :3], d6[..., 3:]
        parent_child = pose_translation[:, 1:, :, 0] - pose_translation[:, self.parents[1:], :, 0]
        a1 = torch.cat([a1[:, :1], parent_child], dim=1)  # (B, num_parts, 3)
        b1 = F.normalize(a1, dim=-1)
        b2 = a2[:, None] - (b1 * a2[:, None]).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)  # (B, num_parts, 3, 3)

    def forward(self, img: torch.tensor, pose_2d: torch.tensor, intrinsic: Optional[torch.tensor],
                z_pose: Optional[torch.tensor]):
        """estimate 3d pose from image and GT 2d pose

        Args:
            img: (B, 3, 128, 128)
            pose_2d: (B, n_parts, 2)
            intrinsic:
            z_pose: latent vector for pose

        Returns:

        """
        intrinsic = torch.tensor(intrinsic, device=img.device)
        inv_intrinsic = torch.inverse(intrinsic)[:, None]
        batchsize = img.shape[0]

        img = (img * 0.5 + 0.5 - self.resnet_mean) / self.resnet_std
        image_feature = self.image_encoder(img)

        # z
        h = self.linear_z1(image_feature)  # (B, 512)
        z = self.linear_z2(h)  # (B, z_dim * 4)

        # pose
        image_feature = self.linear_image(image_feature)  # (B, 512 * n_parts)
        image_feature = image_feature.reshape(batchsize, -1, self.num_bone)
        image_feature = image_feature.permute(2, 0, 1)  # (n_parts, B, -1)

        normalized_pose_2d = pose_2d / (self.image_size / 2) - 1
        encoded_pose = self.positional_encoding(normalized_pose_2d)
        pose_feature = encoded_pose.permute(1, 0, 2)  # (n_parts, B, 4L+2)
        pose_feature = self.linear_pose(pose_feature)  # (n_parts, B, -1)

        if z_pose is None:
            z_pose = torch.randn(batchsize, self.z_dim, device=img.device)
        pose_latent = self.linear_z_pose(z_pose)
        pose_latent = pose_latent.reshape(batchsize, -1, self.num_bone)
        pose_latent = pose_latent.permute(2, 0, 1)  # (n_parts, B, -1)

        # transformer input
        pose_feature = pose_feature + self.pose_positional_embedding + pose_latent + image_feature

        h = self.transformer(pose_feature)  # (n_parts, B, transformer_hidden_dim)
        h = self.linear_out(h)  # (n_parts, B, 7)

        # joint translation
        depth = F.softplus(h[..., 0])  # (n_parts, B)
        pose_homo = torch.cat([pose_2d, torch.ones_like(pose_2d[:, :, :1])], dim=2) * depth.permute(1, 0)[:, :, None]
        pose_translation = torch.matmul(inv_intrinsic,
                                        pose_homo[:, :, :, None])  # (B, n_parts, 3, 1)
        pose_translation = self.scale_pose(pose_translation)

        rot = h[..., 1:]  # (n_parts, B, 6)
        rotation_matrix = self.get_rotation_matrix(rot, pose_translation)  # (B, n_parts, 3, 3)
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


class PretrainedStyleGAN(nn.Module):
    def __init__(self):
        super(PretrainedStyleGAN, self).__init__()
        import sys
        import kornia
        sys.path.append("stylegan2_pytorch")
        from stylegan2_pytorch.model import Generator
        size = 256
        latent = 512
        n_mlp = 8
        channel_multiplier = 2
        device = "cuda"
        ckpt = "stylegan2_pytorch/stylegan2-church-config-f.pt"
        g_ema = Generator(
            size, latent, n_mlp, channel_multiplier=channel_multiplier
        ).to(device)
        checkpoint = torch.load(ckpt)

        g_ema.load_state_dict(checkpoint["g_ema"])
        g_ema.input.input = nn.Parameter(g_ema.input.input[:, :, 1:-1].data)

        self.size = 128
        self.gen = g_ema
        self.cropper = kornia.augmentation.RandomCrop((self.size, self.size), resample='NEAREST')
        self.n_latent = g_ema.n_latent

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor], inject_index):
        z = torch.cat(z, dim=1)
        sample, _ = self.gen([z], inject_index=inject_index)
        if self.training:
            sample = self.cropper(sample)
        else:
            sample = sample[:, :, :, self.size // 2: self.size * 3 // 2]
        return sample, None


class TriNARFGenerator(nn.Module):  # tri-plane nerf
    def __init__(self, config, size, num_bone=1, parent_id=None, num_bone_param=None, black_background=False):
        super(TriNARFGenerator, self).__init__()
        self.config = config
        self.size = size
        self.num_bone = num_bone
        self.ray_sampler = whole_image_grid_ray_sampler
        self.background_ratio = config.background_ratio
        self.black_background = black_background

        z_dim = config.z_dim
        crop_background = config.crop_background

        if config.nerf_params.origin_location == "center+head":
            num_bone_param = num_bone
        self.nerf = TriPlaneNARF(config.nerf_params, z_dim=[z_dim * 2, z_dim], num_bone=num_bone,
                                 bone_length=True,
                                 parent=parent_id, num_bone_param=num_bone_param)
        if not black_background:
            if config.pretrained_background:
                self.background_generator = PretrainedStyleGAN()
            else:
                self.background_generator = StyleGANGenerator(size=size, style_dim=z_dim,
                                                              n_mlp=4, last_channel=3,
                                                              crop_background=crop_background)

    # def init_bg(self):
    #     config = self.config
    #     self.background_generator = StyleGANGenerator(size=self.size, style_dim=config.z_dim,
    #                                                   n_mlp=4, last_channel=3,
    #                                                   crop_background=config.crop_background)
    #     self.background_generator.cuda()
    #     self.nerf.requires_grad_(False)

    def register_canonical_pose(self, pose: np.ndarray):
        self.nerf.register_canonical_pose(pose)

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
                return_intermediate=False, truncation_psi=1, black_bg_if_possible=False, return_disparity=False,
                return_bg=False):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param pose_to_world: wold coordinate of joint
        :param bone_length:
        :param z: latent vector
        :param inv_intrinsics:
        :param return_intermediate:
        :return:
        """
        assert self.num_bone == 1 or (bone_length is not None and pose_to_camera is not None)
        batchsize = pose_to_camera.shape[0]

        grid, homo_img = self.ray_sampler(self.size, self.size, batchsize)

        if not self.black_background:
            z_dim = z.shape[1] // 4
            z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)
        else:
            z_dim = z.shape[1] // 3
            z_for_nerf, z_for_neural_render = torch.split(z, [z_dim * 2, z_dim], dim=1)

        # sparse rendering
        inv_intrinsics = torch.tensor(inv_intrinsics).float().cuda(homo_img.device)
        nerf_output = self.nerf(batchsize, homo_img,
                                pose_to_camera, inv_intrinsics, z_for_nerf,
                                z_for_neural_render,
                                bone_length,
                                Nc=self.config.nerf_params.Nc,
                                Nf=self.config.nerf_params.Nf,
                                return_intermediate=return_intermediate, truncation_psi=truncation_psi,
                                return_disparity=return_disparity)

        fg_color, fg_mask = nerf_output[:2]
        fine_weights = self.nerf.buffers_tensors["fine_weights"]
        fine_depth = self.nerf.buffers_tensors["fine_depth"]

        fg_color = fg_color.reshape(batchsize, 3, self.size, self.size)
        fg_mask = fg_mask.reshape(batchsize, self.size, self.size)

        if not self.black_background and not black_bg_if_possible:
            n_latent = self.background_generator.n_latent
            bg_color, _ = self.background_generator([z_for_background, z_for_neural_render], inject_index=n_latent - 4)
        else:
            bg_color = -1

        rendered_color = fg_color + (1 - fg_mask[:, None]) * bg_color

        if return_intermediate:
            fine_points, fine_density = nerf_output[-1]
            return rendered_color, fg_mask, fine_points, fine_density
        if return_disparity:
            disparity = nerf_output[2]
            disparity = disparity * self.config.nerf_params.coordinate_scale
            return rendered_color, fg_mask, disparity
        if return_bg:
            return fg_color, fg_mask, bg_color
        return rendered_color, fg_mask, fine_weights, fine_depth

    def render_mesh(self, pose_to_camera, intrinsics, z, bone_length, voxel_size=0.003,
                    mesh_th=15, truncation_psi=0.4):
        if not self.black_background:
            z_dim = z.shape[1] // 4
            z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)
        else:
            z_dim = z.shape[1] // 3
            z_for_nerf, z_for_neural_render = torch.split(z, [z_dim * 2, z_dim], dim=1)
        return self.nerf.render_mesh(pose_to_camera, intrinsics, z_for_nerf, z_for_neural_render, bone_length,
                                     voxel_size, mesh_th, truncation_psi, self.size)

    def create_mesh(self, pose_to_camera, z, bone_length, voxel_size=0.003,
                    mesh_th=15, truncation_psi=0.4):
        if not self.black_background:
            z_dim = z.shape[1] // 4
            z_for_nerf, z_for_neural_render, z_for_background = torch.split(z, [z_dim * 2, z_dim, z_dim], dim=1)
        else:
            z_dim = z.shape[1] // 3
            z_for_nerf, z_for_neural_render = torch.split(z, [z_dim * 2, z_dim], dim=1)
        return self.nerf.create_mesh(pose_to_camera, z_for_nerf, z_for_neural_render, bone_length,
                                     voxel_size, mesh_th, truncation_psi)


class SSONARFGenerator(nn.Module):
    def __init__(self, config, size, num_bone=1, parent_id=None, num_bone_param=None):
        super(SSONARFGenerator, self).__init__()
        self.config = config
        self.size = size
        self.num_bone = num_bone
        self.ray_sampler = mask_based_sampler

        nerf = TriPlaneNARF if config.use_triplane else SSONARF

        self.time_conditional = self.config.nerf_params.time_conditional
        self.pose_conditional = self.config.nerf_params.pose_conditional

        z_dim = 0
        if self.time_conditional:
            z_dim += 20
        if self.pose_conditional:
            z_dim += (num_bone - 1) * 9

        if config.nerf_params.origin_location == "center+head":
            num_bone_param = num_bone
        view_dependent = not config.nerf_params.no_ray_direction
        self.nerf = nerf(config.nerf_params, z_dim=z_dim, num_bone=num_bone, bone_length=True,
                         parent=parent_id, num_bone_param=num_bone_param, view_dependent=view_dependent)

    def register_canonical_pose(self, pose: np.ndarray):
        if hasattr(self.nerf, "register_canonical_pose"):
            self.nerf.register_canonical_pose(pose)

    @property
    def memory_cost(self):
        return self.nerf.memory_cost

    @property
    def flops(self):
        return self.nerf.flops

    @staticmethod
    def positional_encoding(x: torch.Tensor, num_frequency: int) -> torch.Tensor:
        """
        positional encoding
        :param x: (B, )
        :param num_frequency: L in nerf paper
        :return:(B, n_freq * 2)
        """
        x = x[:, None] * 2 ** torch.arange(num_frequency, device=x.device) * np.pi
        encoded = torch.cat([torch.cos(x), torch.sin(x)], dim=1)
        return encoded

    @staticmethod
    def pose_encoding(pose: torch.Tensor):
        """

        :param pose: (B, num_bone, 4, 4)
        :return: (B, (num_bone - 1) * 3)
        """
        rot = pose[:, 1:, :3, :3]
        root_rot = pose[:, :1, :3, :3]

        encoded = torch.matmul(root_rot.permute(0, 1, 3, 2), rot)  # (B, num_bone - 1, 3, 3)
        return encoded.reshape(encoded.shape[0], -1)  # (B, (num_bone - 1) * 9)

    def get_latents(self, frame_time: torch.Tensor, pose_to_camera: torch.Tensor):
        zs = []
        if self.time_conditional:
            zs.append(self.positional_encoding(frame_time, num_frequency=10))
        if self.pose_conditional:
            zs.append(self.pose_encoding(pose_to_camera))

        assert len(zs) > 0

        z = torch.cat(zs, dim=1)

        z1 = z2 = z
        return z1, z2

    def forward(self, pose_to_camera, camera_pose, mask, frame_time, bone_length, inv_intrinsics,
                background: Optional[float] = None):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param camera_pose: camera rotation
        :param mask: foreground mask of object, (B, img_size, img_size)
        :param frame_time: normalized time of frame
        :param bone_length:
        :param background: background color
        :param inv_intrinsics:
        :return:
        """
        assert bone_length is not None and pose_to_camera is not None
        assert isinstance(inv_intrinsics, torch.Tensor)
        batchsize = pose_to_camera.shape[0]
        ray_batchsize = self.config.ray_batchsize

        grid, img_coord = self.ray_sampler(mask, ray_batchsize)

        # sparse rendering
        z1, z2 = self.get_latents(frame_time, pose_to_camera)

        # TODO: randomly replace z1 during training
        rendered_color, rendered_mask = self.nerf(batchsize, img_coord,
                                                  pose_to_camera, inv_intrinsics, z1, z2,
                                                  bone_length,
                                                  Nc=self.config.nerf_params.Nc,
                                                  Nf=self.config.nerf_params.Nf,
                                                  return_intermediate=False,
                                                  camera_pose=camera_pose,
                                                  )

        if background is None:
            background = -1
        rendered_color = rendered_color + background * (1 - rendered_mask[:, None])
        return rendered_color, rendered_mask, grid

    def render_entire_img(self, pose_to_camera, inv_intrinsics, frame_time, bone_length,
                          camera_pose=None, render_size=128,
                          semantic_map=False, use_normalized_intrinsics=False, no_grad=True):
        """

        :param pose_to_camera:
        :param inv_intrinsics:
        :param frame_time:
        :param bone_length:
        :param camera_pose:
        :param render_size:
        :param semantic_map:
        :param use_normalized_intrinsics:
        :return:
        """
        # sparse rendering
        z1, z2 = self.get_latents(frame_time, pose_to_camera)
        return self.nerf.render_entire_img(pose_to_camera, inv_intrinsics, z1, z2, bone_length,
                                           camera_pose, render_size, self.config.nerf_params.Nc,
                                           self.config.nerf_params.Nf, semantic_map,
                                           use_normalized_intrinsics, no_grad=no_grad)

    def profile_memory_stats(self, pose_to_camera, inv_intrinsics, frame_time, bone_length,
                             camera_pose=None, render_size=128,
                             semantic_map=False, use_normalized_intrinsics=False):
        """

        :param pose_to_camera:
        :param inv_intrinsics:
        :param frame_time:
        :param bone_length:
        :param camera_pose:
        :param render_size:
        :param semantic_map:
        :param use_normalized_intrinsics:
        :return:
        """
        # sparse rendering
        z1, z2 = self.get_latents(frame_time, pose_to_camera)
        return self.nerf.profile_memory_stats(pose_to_camera, inv_intrinsics, z1, z2, bone_length,
                                              camera_pose, render_size, self.config.nerf_params.Nc,
                                              self.config.nerf_params.Nf, semantic_map,
                                              use_normalized_intrinsics)
