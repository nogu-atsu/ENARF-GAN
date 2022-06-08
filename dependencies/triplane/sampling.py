from typing import Optional

import torch
import torch.nn.functional as F

from cuda_extension.triplane_sampler import triplane_sampler


def sample_feature(tri_plane_features: torch.tensor, position: torch.tensor, reduction: str = "sum", clamp_mask=False,
                   batch_idx: Optional[torch.Tensor] = None):
    """sample tri-plane feature at a position

    :param tri_plane_features: (B, ? * 3, h, w)
    :param position: [-1, 1] in meter, (B, 3, n)
    :param reduction: "prod" or "sum"
    :param clamp_mask: clamp low probabilities
    :param batch_idx: index of data in minibatch

    :return: feature: (B, 32, n)
    """
    batchsize, _, h, w = tri_plane_features.shape
    assert batchsize == 1 or batch_idx is None
    _, _, n = position.shape
    if batchsize == 1 and reduction == "sum":
        position_2d = position.permute(0, 2, 1).contiguous()[:, :, None, :]
        feature = triplane_sampler(tri_plane_features, position_2d)[:, :, :, 0]
    else:
        features = tri_plane_features.reshape(batchsize * 3, -1, h, w)
        # produce 2D coordinate for each tri-plane
        position_2d = position[:, [0, 1, 1, 2, 2, 0]].reshape(batchsize * 3, 2, n)
        position_2d = position_2d.permute(0, 2, 1)[:, :, None]  # (B * 3, n, 1, 2)

        # if batch_idx is not None, place tri-planes side by side to form a single tri-plane (quite tricky)
        if batch_idx is not None:  # transform x coordinate
            actual_batchsize = w // (h + 1)
            scale = 1 / (actual_batchsize * (1 + 1 / h))
            position_2d[:, :, :, 0] = (position_2d[:, :, :, 0] * scale +
                                       batch_idx[None, :, None] * (2 / actual_batchsize) + (scale - 1))

        feature = F.grid_sample(features, position_2d, align_corners=False)
        # feature = torch.cudnn_grid_sampler(features, position_2d)
        feature = feature.reshape(batchsize, 3, -1, n)
        if reduction == "sum":
            feature = feature.sum(dim=1)  # (B, feat_dim, n)
        elif reduction == "prod":
            if clamp_mask:
                feature = (feature.data.clamp(-2, 5) - feature.data) + feature
            feature = torch.sigmoid(feature).prod(dim=1)
        else:
            raise ValueError()
    return feature


def sample_triplane_part_prob(tri_plane_weights: torch.Tensor, position: torch.Tensor,
                              position_validity: torch.Tensor, mode="prod", clamp_mask=False):
    bs, n_bone, _, n = position.shape
    position = position.reshape(bs * n_bone, 3, n)

    # default mode is prod
    if mode == "prod":
        # sample prob from tri-planes and compute product
        weight = sample_feature(tri_plane_weights, position, clamp_mask=clamp_mask,
                                reduction="prod")  # (B * n_bone, 1, n)
        weight = weight.reshape(bs, n_bone, n)
    elif mode == "sum":  # sum and softmax
        weight = sample_feature(tri_plane_weights, position,
                                clamp_mask=clamp_mask, )  # (B * n_bone, 1, n)
        weight = weight.reshape(bs, n_bone, n)

        # # wight for invalid point is 0
        weight = weight - ~position_validity * 1e4
        weight = torch.softmax(weight, dim=1)

    else:
        weight = torch.ones(bs, n_bone, n, device=position.device) / n_bone
    return weight


def sample_weighted_feature_v2(feat_dim: int, tri_plane_features: torch.Tensor, position: torch.Tensor,
                               weight: torch.Tensor, position_validity: torch.Tensor, clamp_mask: bool = False):
    """
    compute weighted feature by sampling from tri-planes for each part
    :param feat_dim:
    :param tri_plane_features:
    :param position:
    :param weight:
    :param position_validity:
    :param clamp_mask:
    :return:
    """
    # only compute necessary elements
    batchsize, n_bone, n = position_validity.shape
    _, ch, tri_size, _ = tri_plane_features.shape

    # place tri-planes side by side to form a single tri-plane (quite tricky)
    feature_padded = F.pad(tri_plane_features, (0, 1))  # (B, ch, 256, 257)
    feature_padded = feature_padded.permute(1, 2, 0, 3).reshape(1, ch, tri_size, (tri_size + 1) * batchsize)

    # gather valid rays
    position_validity = position_validity.reshape(-1)
    assert position_validity.dtype == torch.bool
    valid_args = torch.where(position_validity)[0]  # (num_valid, )
    num_valid = valid_args.shape[0]

    if num_valid > 0:  # num_valid is 3e7 for zju dataset
        position_perm = position.permute(2, 0, 1, 3).reshape(3, batchsize * n_bone * n)  # (3, B * n_bone * n)
        valid_positions = torch.gather(position_perm, dim=1,
                                       index=valid_args[None].expand(3, -1))[None]  # (1, 3, num_valid)
        # challenge: this is very heavy
        value = sample_feature(feature_padded, valid_positions, clamp_mask=clamp_mask,
                               batch_idx=valid_args // (n_bone * n))  # (1, 32, num_valid)
        # gather weight
        weight = torch.gather(weight.reshape(-1), dim=0, index=valid_args)

        # * weight
        value = value * weight[None, None]  # (1, 32, num_valid)

        # memory efficient
        output = torch.zeros(feat_dim, batchsize * n, device=position.device, dtype=torch.float32)
        scatter_idx = valid_args // (n_bone * n) * n + valid_args % n
        output.scatter_add_(dim=1, index=scatter_idx[None].expand(32, -1), src=value.squeeze(0))
        output = output.reshape(feat_dim, batchsize, n).permute(1, 0, 2)
        output = output.contiguous()
    else:
        output = torch.zeros(batchsize, feat_dim, n, device=position.device, dtype=torch.float32)
    return output
