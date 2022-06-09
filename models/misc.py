from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
from torch import nn


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
