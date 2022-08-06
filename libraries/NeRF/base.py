import time
from typing import Optional, Union, List, Dict

import torch
from torch import nn

from libraries.NeRF.activation import MyReLU
from libraries.NeRF.rendering import render, render_entire_img


class NeRFBase(nn.Module):
    def __init__(self, config, z_dim: Union[int, List[int]] = 256,
                 view_dependent: bool = False, **kwargs):
        super(NeRFBase, self).__init__()
        assert hasattr(config, "origin_location")

        self.config = config
        hidden_size = config.hidden_size

        self.origin_location = config.origin_location
        self.coordinate_scale = config.coordinate_scale

        assert self.origin_location in ["center", "center_fixed", "center+head"]

        self.density_activation = MyReLU.apply

        # parameters for position encoding
        nffp = self.config.num_frequency_for_position if "num_frequency_for_position" in self.config else 10
        nffo = self.config.num_frequency_for_other if "num_frequency_for_other" in self.config else 4
        self.num_frequency_for_position = nffp
        self.num_frequency_for_other = nffo

        self.hidden_size = hidden_size

        if type(z_dim) == list:
            self.z_dim = z_dim[0]
            self.z2_dim = z_dim[1]
        else:
            self.z_dim = z_dim
            self.z2_dim = z_dim

        self.view_dependent = view_dependent

        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

        self.temporal_state = {}

    def time_start(self):
        torch.cuda.synchronize()
        start = time.time()
        return start

    def time_end(self, name, start):
        torch.cuda.synchronize()
        end = time.time()
        if name not in self.buffers_tensors:
            self.buffers_tensors[name] = 0
        self.buffers_tensors[name] += end - start

    @property
    def memory_cost(self):
        raise NotImplementedError()

    @property
    def flops(self):
        raise NotImplementedError()

    def calc_density_and_color_from_camera_coord_v2(self, position: torch.Tensor, pose_to_camera: torch.Tensor,
                                                    ray_direction: torch.Tensor, model_input: Dict):
        """compute density from positions in camera coordinate

        :param position: (B, 3, n), n is a very large number of points sampled
        :param pose_to_camera:
        :param ray_direction:
        :param model_input: dictionary of model input
        :return: density of input positions
        """
        raise NotImplementedError()

    def backbone(self, **kwargs):
        raise NotImplementedError()

    def _forward(self, sampled_img_coord, pose_to_camera, inv_intrinsics,
                 render_scale=1, Nc=64, Nf=128, return_intermediate=False,
                 camera_pose: Optional[torch.Tensor] = None,
                 model_input: Dict = {}, return_disparity=False):
        """
        rendering function for sampled rays
        :param sampled_img_coord: sampled image coordinate
        :param pose_to_camera:
        :param inv_intrinsics:
        :param render_scale:
        :param Nc:
        :param Nf:
        :param return_intermediate:
        :param camera_pose:
        :param model_input:
        :param return_disparity:
        :return: color and mask value for sampled rays
        """
        assert not self.view_dependent or camera_pose is not None
        nerf_output = render(self, sampled_img_coord,
                             pose_to_camera,
                             inv_intrinsics,
                             Nc=Nc,
                             Nf=Nf,
                             render_scale=render_scale,
                             return_intermediate=return_intermediate,
                             camera_pose=camera_pose,
                             model_input=model_input)
        if return_intermediate:
            merged_color, merged_mask, _, intermediate_output = nerf_output
            return merged_color, merged_mask, intermediate_output

        merged_color, merged_mask, merged_disparity = nerf_output
        if return_disparity:
            return merged_color, merged_mask, merged_disparity
        return merged_color, merged_mask

    def forward(self, batchsize, sampled_img_coord, pose_to_camera, inv_intrinsics,
                render_scale=1, Nc=64, Nf=128, return_intermediate=False,
                camera_pose: Optional[torch.Tensor] = None,
                return_disparity=False, model_input: Dict = {}):
        """
        rendering function for sampled rays
        :param batchsize:
        :param sampled_img_coord: sampled image coordinate
        :param pose_to_camera:
        :param inv_intrinsics:
        :param render_scale:
        :param Nc:
        :param Nf:
        :param return_intermediate:
        :param camera_pose:
        :param return_disparity:
        :param model_input:
        :return: color and mask value for sampled rays
        """
        return self._forward(sampled_img_coord, pose_to_camera, inv_intrinsics,
                             render_scale, Nc, Nf, return_intermediate,
                             camera_pose, model_input, return_disparity)

    def render_entire_img(self, pose_to_camera, inv_intrinsics, z, z_rend, camera_pose=None,
                          render_size=128, Nc=64, Nf=128, semantic_map=False, use_normalized_intrinsics=False,
                          no_grad=True, model_input={}):
        if self.tri_plane_based:
            model_input["tri_plane_feature"] = self.compute_tri_plane_feature(z)
        return render_entire_img(self, pose_to_camera, inv_intrinsics, camera_pose,
                                 render_size, Nc, Nf, semantic_map, use_normalized_intrinsics,
                                 no_grad, model_input)
