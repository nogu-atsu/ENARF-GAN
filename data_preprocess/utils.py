import numpy as np


def get_bone_length(pose, parents):
    coordinate = pose[:, :3, 3]
    length = np.linalg.norm(coordinate[1:] - coordinate[parents[1:]], axis=-1, keepdims=True)
    return length
