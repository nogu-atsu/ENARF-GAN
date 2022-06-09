from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.structures import Meshes
from torch import nn

from dependencies.NARF.pose_utils import transform_pose
from dependencies.NARF.utils_3d import create_unit_cube_mesh
from dependencies.NeRF_net.utils import in_cube
from dependencies.pytorch3d_utils import compute_projection_matrix_from_inv_intrinsics


def decide_frustrum_range(num_bone, image_coord, pose_to_camera, inv_intrinsics,
                          near_plane, far_plane, return_camera_coord=False):
    assert image_coord.shape[1] == 1
    # update near_plane and far_plane
    joints_z = pose_to_camera[:, :, 2, 3]
    near_plane = torch.clamp_min(joints_z.min() - 3 ** 0.5, near_plane)
    far_plane = torch.clamp_min(joints_z.max() + 3 ** 0.5, far_plane)
    n_samples_to_decide_depth_range = 32
    batchsize, _, _, n = image_coord.shape
    with torch.no_grad():
        # rotation & translation
        R = pose_to_camera[:, :, :3, :3]  # (B, n_bone, 3, 3)
        t = pose_to_camera[:, :, :3, 3:]  # (B, n_bone, 3, 1)

        if inv_intrinsics.ndim == 2:
            image_coord = image_coord.reshape(batchsize, 3, n)
            # img coord -> camera coord
            sampled_camera_coord = torch.matmul(inv_intrinsics, image_coord)
        else:
            # reshape for multiplying inv_intrinsics
            image_coord = image_coord.reshape(batchsize, 3, n)

            # img coord -> camera coord
            sampled_camera_coord = torch.matmul(inv_intrinsics, image_coord)
            sampled_camera_coord = sampled_camera_coord.reshape(batchsize, 3, n)

        # ray direction in camera coordinate
        ray_direction = sampled_camera_coord  # (B, 3, n)

        # sample points to decide depth range
        sampled_depth = torch.linspace(near_plane, far_plane, n_samples_to_decide_depth_range, device="cuda")
        sampled_points_on_rays = ray_direction[:, :, :, None] * sampled_depth

        # camera coord -> bone coord
        sampled_points_on_rays = torch.matmul(
            R.permute(0, 1, 3, 2),
            sampled_points_on_rays.reshape(batchsize, 1, 3, -1) - t)  # (B, n_bone, 3, n * n_samples)
        sampled_points_on_rays = sampled_points_on_rays.reshape(batchsize * num_bone, 3, n,
                                                                n_samples_to_decide_depth_range)
        # inside the cube [-1, 1]^3?
        inside = in_cube(sampled_points_on_rays)  # (B*num_bone, 1, n, n_samples_to_decide_depth_range)

        # minimum-maximum depth
        large_value = 1e3
        depth_min = torch.where(inside, sampled_depth,
                                torch.full_like(inside.float(), large_value)).min(dim=3)[0]
        depth_max = torch.where(inside, sampled_depth,
                                torch.full_like(inside.float(), -large_value)).max(dim=3)[0]

        # adopt the smallest/largest values among bones
        depth_min = depth_min.reshape(batchsize, num_bone, 1, n).min(dim=1, keepdim=True)[0]  # B x 1 x 1 x n
        depth_max = depth_max.reshape(batchsize, num_bone, 1, n).max(dim=1, keepdim=True)[0]

        # # replace values if no intersection
        validity = depth_min != large_value  # valid ray
        depth_min = torch.where(depth_min != large_value, depth_min, torch.full_like(depth_min, near_plane))
        depth_max = torch.where(depth_max != -large_value, depth_max, torch.full_like(depth_max, far_plane))

        depth_min = torch.clamp_min(depth_min, near_plane)

        if return_camera_coord:
            return depth_min, depth_max, ray_direction, validity

        # camera coord -> bone coord
        ray_direction = torch.matmul(R.permute(0, 1, 3, 2), ray_direction[:, None])
        ray_direction = ray_direction.reshape(batchsize * num_bone, 3, n)
        camera_origin = -torch.matmul(R.permute(0, 1, 3, 2), t)  # (B, num_bone, 3, 1)
        camera_origin = camera_origin.reshape(batchsize * num_bone, 3, 1).expand(batchsize * num_bone, 3, n)
        return camera_origin, depth_min, depth_max, ray_direction


def coarse_sample(num_bone: int, image_coord: torch.tensor, pose_to_camera: torch.tensor,
                  inv_intrinsics: torch.tensor, near_plane: float = 0.3, far_plane: float = 5,
                  Nc: int = 64, camera_pose: Optional[torch.Tensor] = None, view_dependent=False,
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                             Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                             Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    batchsize, _, _, n = image_coord.shape
    with torch.no_grad():
        (depth_min, depth_max,
         ray_direction, ray_validity) = decide_frustrum_range(num_bone, image_coord,
                                                              pose_to_camera,
                                                              inv_intrinsics, near_plane,
                                                              far_plane, return_camera_coord=True)
        if batchsize == 1:  # remove invalid rays
            depth_min = depth_min[:, :, :, ray_validity[0, 0, 0]]
            depth_max = depth_max[:, :, :, ray_validity[0, 0, 0]]
            ray_direction = ray_direction[:, :, ray_validity[0, 0, 0]]

        if depth_min.shape[-1] > 0:
            n = depth_min.shape[-1]
            if view_dependent:
                ray_direction_in_world = F.normalize(ray_direction, dim=1)
                ray_direction_in_world = torch.matmul(camera_pose.permute(0, 2, 1), ray_direction_in_world)
            else:
                ray_direction_in_world = None

            depth_min = depth_min.squeeze(1)
            depth_max = depth_max.squeeze(1)
            start = depth_min * ray_direction  # (B, 3, n)
            end = depth_max * ray_direction  # (B, 3, n)
            # coarse ray sampling
            bins = torch.linspace(0, 1, Nc + 1, dtype=torch.float, device="cuda").reshape(1, 1, 1, Nc + 1)
            coarse_depth = (depth_min.unsqueeze(-1) * (1 - bins) +
                            depth_max.unsqueeze(-1) * bins)  # (B, 1, n, Nc + 1)

            coarse_points = start.unsqueeze(-1) * (1 - bins) + end.unsqueeze(-1) * bins  # (B, 3, n, (Nc+1))
            coarse_points = (coarse_points[:, :, :, 1:] + coarse_points[:, :, :, :-1]) / 2
            coarse_points = coarse_points.reshape(batchsize, 3, -1)
            return (coarse_depth, ray_direction_in_world, coarse_points,
                    ray_validity, n, depth_min, depth_max, start, end)
        else:
            return (None,) * 9


def coarse_sample_v2(self, pose_to_camera, inv_intrinsics, img_size):
    # TODO: compute only once
    device = pose_to_camera.device
    batch_size = pose_to_camera.shape[0]
    num_bone = pose_to_camera.shape[1]
    verts, faces = create_unit_cube_mesh(device=device)
    with torch.no_grad():
        # rotation & translation
        R = pose_to_camera[:, :, :3, :3]  # (B, n_bone, 3, 3)
        t = pose_to_camera[:, :, :3, 3:]  # (B, n_bone, 3, 1)

        verts = torch.matmul(R, verts[None, None]) + t  # (B, n_bone, 3, 8)
        verts = verts.permute(0, 1, 3, 2).reshape(batch_size * num_bone, 8, 3)
        faces = faces[None].expand(batch_size * num_bone, 12, 3)  # (B * n_bone, 12, 3)
        projection_matrix = compute_projection_matrix_from_inv_intrinsics(inv_intrinsics, img_size)
        cubes = Meshes(verts=verts, faces=faces)
        cameras = FoVPerspectiveCameras(device=device, K=projection_matrix)
        raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=1e-11,
            faces_per_pixel=2,
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        zbuf = rasterizer(cubes).zbuf
        depth_max = zbuf[:, :, :, 1].reshape(batch_size, num_bone,
                                             img_size, img_size).max(dim=1)[0]  # (B, img_size, img_size)
        depth_min = zbuf[:, :, :, 0]
        depth_min = depth_min.masked_fill(depth_min < 0, 1e9)
        depth_min = depth_min.reshape(batch_size, num_bone,
                                      img_size, img_size).min(dim=1)[0]  # (B, img_size, img_size)
        depth_min = depth_min.masked_fill(depth_min > 1e8, -1)

        # save images for debug
        import matplotlib.pyplot as plt
        depth_max_img = depth_max.reshape(batch_size * img_size, img_size).cpu().numpy()
        depth_min_img = depth_min.reshape(batch_size * img_size, img_size).cpu().numpy()
        plt.subplot(3, 1, 1)
        plt.imshow(depth_max_img)
        plt.subplot(3, 1, 2)
        plt.imshow(depth_min_img)
        plt.subplot(3, 1, 3)
        plt.imshow((depth_max_img - depth_min_img) / self.coordinate_scale)
        plt.colorbar()
        # plt.savefig('depth_max_min.png')
    return depth_max, depth_min


def coarse_to_fine_sample(narf: nn.Module, image_coord: torch.tensor, pose_to_camera: torch.tensor,
                          inv_intrinsics: torch.tensor, z: torch.tensor, z_rend: torch.tensor,
                          bone_length: torch.tensor = None, near_plane: float = 0.3, far_plane: float = 5,
                          Nc: int = 64, Nf: int = 128, render_scale: float = 1,
                          camera_pose: Optional[torch.Tensor] = None
                          ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
                                     Optional[torch.Tensor], Optional[torch.Tensor]]:
    """

    :param narf: narf module
    :param image_coord:
    :param pose_to_camera:
    :param inv_intrinsics:
    :param z: z for NeRF, tri-plane for TriNeRF
    :param z_rend:
    :param bone_length:
    :param near_plane:
    :param far_plane:
    :param Nc:
    :param Nf:
    :param render_scale:
    :param camera_pose:
    :return:
    """
    batchsize, _, _, _ = image_coord.shape
    num_bone = narf.num_bone
    view_dependent = narf.view_dependent

    (coarse_depth, ray_direction_in_world, coarse_points,
     ray_validity, n, depth_min, depth_max, start, end) = coarse_sample(num_bone, image_coord, pose_to_camera,
                                                                        inv_intrinsics, near_plane, far_plane,
                                                                        Nc, camera_pose,
                                                                        view_dependent=view_dependent)
    if coarse_depth is None:
        return (None,) * 4

    coarse_density, _ = narf.calc_density_and_color_from_camera_coord(coarse_points, pose_to_camera,
                                                                      bone_length,
                                                                      z, z_rend, ray_direction=None)

    Np = coarse_depth.shape[-1]  # Nc or Nc + 1
    # calculate weight for fine sampling
    coarse_density = coarse_density.reshape(batchsize, 1, -1, Nc)[:, :, :, :Np - 1]
    # # delta = distance between adjacent samples
    delta = coarse_depth[:, :, :, 1:] - coarse_depth[:, :, :, :-1]  # B x 1 x n x Np - 1

    density_delta = coarse_density * delta * render_scale
    T_i = torch.exp(-(torch.cumsum(density_delta, dim=3) - density_delta))
    weights = T_i * (1 - torch.exp(-density_delta))  # B x 1 x n x Np-1
    weights = weights.reshape(-1, Np - 1)

    # fine ray sampling
    weights = F.pad(weights, (1, 1, 0, 0))
    weights = (torch.maximum(weights[:, :-2], weights[:, 1:-1]) +
               torch.maximum(weights[:, 1:-1], weights[:, 2:])) / 2 + 0.01

    bins = (torch.multinomial(weights,
                              Nf, replacement=True).reshape(batchsize, 1, -1, Nf).float() / Nc +
            torch.cuda.FloatTensor(batchsize, 1, n, Nf).uniform_() / Nc)

    # sort points
    bins = torch.sort(bins, dim=-1)[0]
    fine_depth = depth_min.unsqueeze(3) * (1 - bins) + depth_max.unsqueeze(3) * bins  # (B, 1, n, Nf)

    fine_points = start.unsqueeze(3) * (1 - bins) + end.unsqueeze(3) * bins  # (B, 3, n, Nf)

    fine_points = fine_points.reshape(batchsize, 3, -1)

    if not narf.training:
        narf.temporal_state.update({
            "coarse_density": coarse_density,
            "coarse_T_i": T_i,
            "coarse_weights": weights,
            "coarse_depth": coarse_depth,
            "fine_depth": fine_depth,
            "fine_points": fine_points,
            "near_plane": near_plane,
            "far_plane": far_plane
        })

    if pose_to_camera.requires_grad:
        raise NotImplementedError("Currently pose should not be differentiable")

    return (
        fine_depth,  # (B, 1, n, Nf)
        fine_points,  # (B, 3, n*Nf),
        ray_direction_in_world,  # (B, 3, n) or None
        ray_validity,  # (B, 1, 1, n)
    )


def render(narf: nn.Module, image_coord: torch.tensor, pose_to_camera: torch.tensor,
           inv_intrinsics: torch.tensor, z: torch.tensor, z_rend: torch.tensor,
           bone_length: torch.tensor, thres: float = 0.0, render_scale: float = 1, Nc: int = 64,
           Nf: int = 128, semantic_map: bool = False, return_intermediate: bool = False,
           camera_pose: Optional[torch.Tensor] = None, tri_plane_feature: Optional[torch.Tensor] = None,
           truncation_psi=1) -> (torch.tensor,) * 3:
    near_plane = 0.3
    # n <- number of sampled pixels
    # image_coord: B x groups x 3 x n
    # camera_extrinsics: B x 4 x 4
    # camera_intrinsics: 3 x 3

    batchsize, num_bone, _, n = image_coord.shape
    device = image_coord.device

    pose_to_camera, bone_length = transform_pose(pose_to_camera, bone_length,
                                                 narf.origin_location, narf.parent_id)

    if narf.coordinate_scale != 1:
        pose_to_camera[:, :, :3, 3] *= narf.coordinate_scale

    # narf.coarse_sample_v2(pose_to_camera, inv_intrinsics, 512)

    if not hasattr(narf, "buffers_tensors"):
        narf.buffers_tensors = {}

    if narf.tri_plane_based:
        if tri_plane_feature is None:
            z = narf.compute_tri_plane_feature(z, bone_length, truncation_psi)
        else:
            z = tri_plane_feature
        narf.buffers_tensors["tri_plane_feature"] = z
        if not narf.training:
            narf.temporal_state["tri_plane_feature"] = z

    (fine_depth, fine_points,
     ray_direction_in_world,
     ray_validity) = coarse_to_fine_sample(narf, image_coord, pose_to_camera,
                                           inv_intrinsics,
                                           z=z,
                                           z_rend=z_rend,
                                           bone_length=bone_length,
                                           near_plane=near_plane, Nc=Nc,
                                           Nf=Nf,
                                           render_scale=render_scale,
                                           camera_pose=camera_pose)

    # narf.valid_rays += 0 if ray_validity is None else ray_validity[0, 0, 0].sum()
    if fine_depth is not None:
        # fine density & color # B x groups x 1 x n*(Nc+Nf), B x groups x 3 x n*(Nc+Nf)
        if semantic_map and narf.config.mask_input:
            narf.save_mask = True
        (fine_density,
         fine_color) = narf.calc_density_and_color_from_camera_coord(fine_points, pose_to_camera,
                                                                     bone_length,
                                                                     z, z_rend,
                                                                     ray_direction=ray_direction_in_world)

        narf.buffers_tensors["fine_density"] = fine_density

        Np = fine_depth.shape[-1]  # Nf

        if return_intermediate:
            intermediate_output = (fine_points, fine_density)

        if semantic_map and narf.config.mask_input:
            narf.save_mask = False

        # semantic map
        if semantic_map:
            assert False, "semantic map rendering will be implemented later"
            bone_idx = torch.arange(num_bone).cuda()
            seg_color = torch.stack([bone_idx // 9, (bone_idx // 3) % 3, bone_idx % 3], dim=1) - 1  # num_bone x 3
            seg_color[::2] = seg_color.flip(dims=(0,))[1 - num_bone % 2::2]  # num_bone x 3
            fine_color = seg_color[narf.mask_prob.reshape(-1)]
            fine_color = fine_color.reshape(batchsize, 1, -1, 3).permute(0, 1, 3, 2)
            fine_color = fine_color.reshape(batchsize, 1, 3, n, -1)[:, :, :, :, :Np - 1]
        else:
            fine_color = fine_color.reshape(batchsize, 3, -1, Np)[:, :, :, :Np - 1]

        fine_density = fine_density.reshape(batchsize, 1, -1, Np)[:, :, :, :Np - 1]

        sum_fine_density = fine_density

        # if thres > 0:
        #     # density = inf if density exceeds thres
        #     sum_fine_density = (sum_fine_density > thres) * 100000

        delta = fine_depth[:, :, :, 1:] - fine_depth[:, :, :, :-1]  # B x 1 x n x Np-1
        sum_density_delta: torch.Tensor = sum_fine_density * delta * render_scale  # B x 1 x n x Np-1

        T_i = torch.exp(-(torch.cumsum(sum_density_delta, dim=3) - sum_density_delta))
        weights = T_i * (1 - torch.exp(-sum_density_delta))  # B x 1 x n x Nc+Nf-1

        narf.buffers_tensors["fine_weights"] = weights
        narf.buffers_tensors["fine_depth"] = fine_depth

        if not narf.training:
            narf.temporal_state["fine_density"] = fine_density
            narf.temporal_state["fine_T_i"] = T_i
            narf.temporal_state["fine_cum_weight"] = torch.cumsum(weights, dim=-1)

        fine_depth = fine_depth.reshape(batchsize, 1, -1, Np)[:, :, :, :-1]

        rendered_color = torch.sum(weights * fine_color, dim=3).squeeze(1)  # B x 3 x n
        rendered_mask = torch.sum(weights, dim=3).reshape(batchsize, -1)  # B x n
        rendered_disparity = torch.sum(weights * 1 / fine_depth, dim=3).reshape(batchsize, -1)  # B x n

        if batchsize == 1:  # revert invalid rays

            def revert_invalid_rays(color, mask, disparity):
                _color = torch.zeros(batchsize, 3, n, device=device)
                _color[:, :, ray_validity[0, 0, 0]] = color
                _mask = torch.zeros(batchsize, n, device=device)
                _mask[:, ray_validity[0, 0, 0]] = mask
                _disparity = torch.zeros(batchsize, n, device=device)
                _disparity[:, ray_validity[0, 0, 0]] = disparity
                return _color, _mask, _disparity

            rendered_color, rendered_mask, rendered_disparity = revert_invalid_rays(rendered_color,
                                                                                    rendered_mask,
                                                                                    rendered_disparity)
    else:
        rendered_color = torch.zeros(batchsize, 3, n, device=device)
        rendered_mask = torch.zeros(batchsize, n, device=device)
        rendered_disparity = torch.zeros(batchsize, n, device=device)

    if return_intermediate:
        return rendered_color, rendered_mask, rendered_disparity, intermediate_output

    return rendered_color, rendered_mask, rendered_disparity


def render_entire_img(narf, pose_to_camera, inv_intrinsics, z, z_rend, bone_length, camera_pose=None,
                      render_size=128, Nc=64, Nf=128,
                      semantic_map=False, use_normalized_intrinsics=False, no_grad=True):
    """

    :param pose_to_camera:
    :param inv_intrinsics:
    :param z:
    :param z_rend:
    :param bone_length:
    :param camera_pose:
    :param render_size:
    :param Nc:
    :param Nf:
    :param semantic_map:
    :param use_normalized_intrinsics:
    :return:
    """
    assert z is None or z.shape[0] == 1
    assert bone_length is None or bone_length.shape[0] == 1
    ray_batchsize = narf.config.render_bs
    if use_normalized_intrinsics:
        img_coord = torch.stack([(torch.arange(render_size * render_size) % render_size + 0.5) / render_size,
                                 (torch.arange(render_size * render_size) // render_size + 0.5) / render_size,
                                 torch.ones(render_size * render_size).long()], dim=0).float()
    else:
        img_coord = torch.stack([torch.arange(render_size * render_size) % render_size + 0.5,
                                 torch.arange(render_size * render_size) // render_size + 0.5,
                                 torch.ones(render_size * render_size).long()], dim=0).float()

    img_coord = img_coord[None, None].cuda()

    rendered_color = []
    rendered_mask = []
    rendered_disparity = []

    with torch.set_grad_enabled(not no_grad):
        if narf.tri_plane_based:
            if narf.origin_location == "center+head":
                _bone_length = torch.cat([bone_length,
                                          torch.ones(bone_length.shape[0], 1, 1, device=bone_length.device)],
                                         dim=1)  # (B, 24)
            else:
                _bone_length = bone_length
            tri_plane_feature = narf.compute_tri_plane_feature(z, _bone_length)
        else:
            tri_plane_feature = None

        for i in range(0, render_size ** 2, ray_batchsize):
            (rendered_color_i, rendered_mask_i,
             rendered_disparity_i) = render(narf, img_coord[:, :, :, i:i + ray_batchsize],
                                            pose_to_camera[:1],
                                            inv_intrinsics,
                                            z=z,
                                            z_rend=z_rend,
                                            bone_length=bone_length,
                                            Nc=Nc,
                                            Nf=Nf,
                                            camera_pose=camera_pose,
                                            tri_plane_feature=tri_plane_feature)
            rendered_color.append(rendered_color_i)
            rendered_mask.append(rendered_mask_i)
            rendered_disparity.append(rendered_disparity_i)

        rendered_color = torch.cat(rendered_color, dim=2)
        rendered_mask = torch.cat(rendered_mask, dim=1)
        rendered_disparity = torch.cat(rendered_disparity, dim=1)

    return (rendered_color.reshape(3, render_size, render_size),  # 3 x size x size
            rendered_mask.reshape(render_size, render_size),  # size x size
            rendered_disparity.reshape(render_size, render_size))  # size x size
