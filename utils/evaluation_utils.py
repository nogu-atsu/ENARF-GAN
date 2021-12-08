import numpy as np


def rigid_transform_3D(A, B):
    """compute rigid transform between A and B

    Args:
        A: (B, n_parts, 3, 1)
        B: (B, n_parts, 3, 1)

    Returns:
        rotation and translation matrix with shape (B, 3, 3) and (B, 3, 1)?
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, centroid_A) + centroid_B
    return R, t


def rigid_align(A, B):
    """compute aligned A

    Args:
        A: (B, n_parts, 3)
        B: (B, n_parts, 3)

    Returns:
        aligned A
    """
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2


if __name__ == "__main__":
    A = np.ones((20, 3))
    B = np.ones((20, 3))
    r, t = rigid_transform_3D(A, B)
    print(r.shape, t.shape)
