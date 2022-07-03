import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    Textures
)
from pytorch3d.structures import Meshes
from tqdm import tqdm

from dependencies.NARF.pose_utils import transform_pose
from dependencies.pytorch3d_utils import compute_projection_matrix_from_intrinsics


def render_mesh_(meshes, intrinsics, img_size, render_size=512):
    device = intrinsics.device
    (vertices, triangles, textures) = meshes
    meshes = Meshes(verts=[vertices], faces=[triangles], textures=textures)

    projection_matrix = compute_projection_matrix_from_intrinsics(intrinsics, img_size)
    cameras = FoVPerspectiveCameras(device=device, K=projection_matrix)
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

    raster_settings = RasterizationSettings(
        image_size=render_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(meshes)
    images = images[0, :, :, :3]
    images = (images.cpu().numpy()[::-1, ::-1] * 255).astype("uint8")

    return images


def create_mesh(self, pose_to_camera, center, voxel_size=0.003,
                mesh_th=15, model_input={}):
    import mcubes

    ray_batchsize = self.config.render_bs if hasattr(self.config, "render_bs") else 1048576
    device = pose_to_camera.device
    cube_size = int(1 / voxel_size)

    bins = torch.arange(-cube_size, cube_size + 1) / cube_size
    p = (torch.stack(torch.meshgrid(bins, bins, bins)).reshape(1, 3, -1) + center.cpu()) * self.coordinate_scale

    if self.coordinate_scale != 1:
        pose_to_camera[:, :, :3, 3] *= self.coordinate_scale

    density = []
    for i in tqdm(range(0, p.shape[-1], ray_batchsize)):
        _density = self.calc_density_and_color_from_camera_coord_v2(
            p[:, :, i:i + ray_batchsize].cuda(non_blocking=True),
            pose_to_camera, ray_direction=None,
            model_input=model_input)[0]  # (1, 1, n)
        density.append(_density)
    density = torch.cat(density, dim=-1)
    density = density.reshape(cube_size * 2 + 1, cube_size * 2 + 1, cube_size * 2 + 1).cpu().numpy()

    vertices, triangles = mcubes.marching_cubes(density, mesh_th)
    vertices = (vertices - cube_size) * voxel_size  # (V, 3)
    vertices = torch.tensor(vertices, device=device).float() + center[:, :, 0]
    triangles = torch.tensor(triangles.astype("int64")).to(device)

    verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb)
    return (vertices, triangles, textures)
