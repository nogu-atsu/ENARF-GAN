import torch


def compute_projection_matrix(min_x, max_x, min_y, max_y, znear, zfar, device) -> torch.Tensor:
    """
    Compute the calibration matrix K of shape (N, 4, 4)
    modified from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html#FoVPerspectiveCameras.compute_projection_matrix

    Args:
        min_x: left of fov at z=1
        max_x: right of fov at z=1
        min_y: top of fov at z=1
        max_y: bottom of fov at z=1
        znear: near clipping plane of the view frustrum.
        zfar: far clipping plane of the view frustrum.
        device: device

    Returns:
        torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
    """
    N = min_x.shape[0]  # batchsize
    K = torch.zeros((N, 4, 4), device=device, dtype=torch.float32)
    ones = torch.ones((N), dtype=torch.float32, device=device)

    # NOTE: In OpenGL the projection matrix changes the handedness of the
    # coordinate frame. i.e the NDC space positive z direction is the
    # camera space negative z direction. This is because the sign of the z
    # in the projection matrix is set to -1.0.
    # In pytorch3d we maintain a right handed coordinate system throughout
    # so the so the z sign is 1.0.
    z_sign = 1.0

    K[:, 0, 0] = 2.0 / (max_x - min_x)
    K[:, 1, 1] = 2.0 / (max_y - min_y)
    K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
    K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
    K[:, 3, 2] = z_sign * ones

    # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
    # is at the near clipping plane and z = 1 when the point is at the far
    # clipping plane.
    K[:, 2, 2] = z_sign * zfar / (zfar - znear)
    K[:, 2, 3] = -(zfar * znear) / (zfar - znear)

    return K


def compute_projection_matrix_from_intrinsics(K, size):
    """
    Compute the calibration matrix K of shape (N, 4, 4)
    :param K: intrinsics, (N, 3, 3)
    :param size: image size, int
    :return:
    """
    max_x = K[:, 0, 2] / K[:, 0, 0]
    max_y = K[:, 1, 2] / K[:, 1, 1]
    min_x = -(size - K[:, 0, 2]) / K[:, 0, 0]
    min_y = -(size - K[:, 1, 2]) / K[:, 1, 1]
    device = K.device
    return compute_projection_matrix(min_x, max_x, min_y, max_y,
                                     znear=1.0, zfar=100.0, device=device)
