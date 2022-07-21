import numpy as np

SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13,
                         14, 16, 17, 18, 19, 20, 21])


def get_bone_length(pose, parents):
    coordinate = pose[:, :3, 3]
    length = np.linalg.norm(coordinate[1:] - coordinate[parents[1:]], axis=-1, keepdims=True)
    return length
