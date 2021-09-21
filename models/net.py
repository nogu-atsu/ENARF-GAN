import numpy as np
import torch
import torchvision
from torch import nn

from NARF.models.net import get_nerf_module
from NARF.models.tiny_utils import whole_image_grid_ray_sampler
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


class Encoder(nn.Module):
    def __init__(self, parents: np.ndarray, intrinsic: np.ndarray):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        # remove MaxPool, AvgPool, and linear
        self.image_encoder = nn.Sequential(*[l for l in resnet.children() if not isinstance(l, nn.MaxPool2d)][:-2])
        self.image_feat_dim = 512
        self.image_size = 128
        self.z_dim = 256

        transformer_hidden_dim = 256
        transformer_n_head = 4
        self.transformer = nn.Transformer(d_model=transformer_hidden_dim, nhead=transformer_n_head,
                                          num_encoder_layers=2, num_decoder_layers=2,
                                          dim_feedforward=transformer_hidden_dim)

        self.linear_image = nn.Linear(self.image_feat_dim, transformer_hidden_dim)
        self.linear_pose = nn.Linear(2, transformer_hidden_dim)
        self.linear_out = nn.Linear(transformer_hidden_dim, 7)

        # z
        self.conv_z = nn.Conv2d(self.image_feat_dim, self.z_dim * 4, 3, 2, 1)
        self.linear_z = nn.Linear(self.z_dim * 4, self.z_dim * 4)

        self.intrinsic = intrinsic.astype("float32")
        self.parents = parents  # parent ids
        self.mean_bone_length = 0.15  # mean length of bone. Should be calculated from dataset??

    def scale_pose(self, pose):
        bone_length = torch.linalg.norm(pose[:, 1:] - pose[:, self.parents[1:]], dim=2)[:, :, 0]  # (B, n_parts-1)
        mean_bone_length = bone_length.mean(dim=1)
        return pose / mean_bone_length * self.mean_bone_length

    def forward(self, img: torch.tensor, pose_2d: torch.tensor):
        """estimate 3d pose from image and GT 2d pose

        Args:
            img: (B, 3, 128, 128)
            pose_2d: (B, n_parts, 2)

        Returns:

        """
        intrinsic = torch.tensor(self.intrinsic, device=img.device)
        inv_intrinsic = torch.inverse(intrinsic)
        batchsize = img.shape[0]
        feature_size = self.image_size // 16
        image_feature = self.image_encoder(img)

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

        h = self.transformer(image_feature, pose_feature)  # (n_parts, B, transformer_hidden_dim)
        h = self.linear_out(h)  # (n_parts, B, 7)

        rot = h[..., :6]  # (n_parts, B, 6)
        depth = h[..., 6]  # (n_parts, B)

        rotation_matrix = rotation_6d_to_matrix(rot)  # (n_parts, B, 3, 3)
        rotation_matrix = rotation_matrix.permute(1, 0, 2, 3)

        pose_homo = torch.cat([pose_2d, depth.transpose()[:, :, None]])
        pose_translation = torch.matmul(inv_intrinsic, pose_homo[:, :, :, None])  # (B, n_parts, 3, 1)
        pose_translation = self.scale_pose(pose_translation)

        pose_Rt = torch.cat([rotation_matrix, pose_translation], dim=-1)  # (B, n_parts, 3, 4)
        num_parts = pose_Rt.shape[1]
        pose_Rt = torch.cat([pose_Rt,
                             torch.tensor([0, 0, 0, 1])[None, None, None].expand(batchsize, num_parts, 1, 4)],
                            dim=2)

        # bone length
        coordinate = pose_Rt[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parents[1:]], axis=1)
        bone_length = length[:, None]

        bone_mask = None  # TODO

        return pose_Rt, z, bone_length, bone_mask
