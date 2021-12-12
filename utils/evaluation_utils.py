import numpy as np


def rigid_transform_3D(A, B):
    """compute rigid transform between A and B

    Args:
        A: (B, n_parts, 3)
        B: (B, n_parts, 3)

    Returns:
        rotation and translation matrix with shape (B, 1, 3, 3) and (B, 1, 3, 1)
    """
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # isomorphic scaling
    scale_A = np.linalg.norm(A - centroid_A, axis=-1).mean(axis=-1)
    scale_B = np.linalg.norm(B - centroid_B, axis=-1).mean(axis=-1)
    scale = scale_B / (scale_A + 1e-8)

    H = np.matmul((A - centroid_A).transpose(0, 2, 1) * scale[:, None, None], B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.matmul(V.transpose(0, 2, 1), U.transpose(0, 2, 1))

    V[:, 2] = V[:, 2] * np.linalg.det(R)[:, None]

    R = np.matmul(V.transpose(0, 2, 1), U.transpose(0, 2, 1)) * scale[:, None, None]

    t = -np.matmul(R[:, None], centroid_A[:, :, :, None]) + centroid_B[:, :, :, None]
    return R[:, None], t


def rigid_align(A, B):
    """compute aligned A

    Args:
        A: (B, n_parts, 3, 1)
        B: (B, n_parts, 3, 1)

    Returns:
        aligned A
    """
    R, t = rigid_transform_3D(A.squeeze(-1), B.squeeze(-1))
    A2 = np.matmul(R, A) + t
    return A2


def pampjpe(estim: np.ndarray, gt: np.ndarray, data_scale: float = 2000.0):
    """PAMPJPE

    Args:
        estim: (B, n_parts, 3, 1)
        gt: (B, n_parts, 3, 1)
        data_scale: scale inversion factor of GT pose

    Returns:
        PAMPJPE
    """
    gt = gt * data_scale
    aligned_estim = rigid_align(estim, gt)
    metric = np.linalg.norm(aligned_estim - gt, axis=2).mean()
    return metric


if __name__ == "__main__":
    A = np.ones((3, 20, 3))
    B = np.ones((3, 20, 3))
    r, t = rigid_transform_3D(A, B)
    print(r.shape, t.shape)

    A = np.random.randn(3, 20, 3, 1)
    B = -3 * A + 1
    A2 = rigid_align(A, B)
    print(A2.shape)

    mpjpe = np.linalg.norm(A2 - B, axis=2).mean()
    print(mpjpe)
