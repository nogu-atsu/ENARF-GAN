import numpy as np
import torch


def pose_to_image_coord(pose_to_camera, intrinsics):
    image_coord = pose_to_camera[:, :3, 3]
    image_coord = image_coord / image_coord[:, 2:3]  # devide by z
    image_coord = image_coord.transpose()[None]  # 1 x 3 x num_joints
    image_coord = np.matmul(intrinsics, image_coord)
    return image_coord


def create_mask(hpp, joint_mat_camera, joint_pos_image, size, thickness=1.5):
    # For the first one data in minibatch only
    # draw bones
    a = joint_pos_image[0, :2, 1:].transpose(1, 0)  # end point, num_joint x 2
    b = joint_pos_image[0, :2, hpp.prev_seq[1:]]  # start point, num_joint x 2

    camera_pos_a = joint_mat_camera[0, 1:, :3, 3]
    camera_pos_b = joint_mat_camera[0, hpp.prev_seq[1:], :3, 3]

    x, y = np.meshgrid(np.arange(size), np.arange(size))
    c = np.stack([x, y], axis=2).reshape(-1, 2)  # xy coordinate of each pixel

    ab = b - a  # len(a) x 2
    ac = c[None] - a[:, None]  # len(a) x size**2 x 2
    acab = np.matmul(ac, ab[:, :, None]).squeeze(2)  # len(a) x size**2

    abab = (ab ** 2).sum(axis=1)[:, None]  # len(a) x 1
    acac = (ac ** 2).sum(axis=2)  # len(a) x size**2
    mask = (0 <= acab) * (acab <= abab) * (acab ** 2 >= abab * (acac - thickness ** 2)) * (abab > 1e-8)
    s = acab / (abab + 1e-10)  # len(a) x size**2, clip around [0, 1]

    camera_z_a = camera_pos_a[:, 2]
    camera_z_b = camera_pos_b[:, 2]
    t = s * camera_z_a[:, None] / (s * camera_z_a[:, None] + (1 - s) * camera_z_b[:, None])  # len(a) x size**2
    camera_z_c = camera_z_a[:, None] * (1 - t) + camera_z_b[:, None] * t

    part_bone_disparity = 1 / (camera_z_c + 1e-8) * mask
    camera_disparity_c = part_bone_disparity.max(axis=0)

    mask = np.clip(mask.sum(axis=0), 0, 1).reshape(size, size)
    camera_disparity = camera_disparity_c
    camera_disparity = camera_disparity.reshape(size, size)

    # part-wise bone disparity
    bone_idx = np.array([hpp.prev_seq[idx] if hpp.is_blank[idx] else idx for idx in hpp.prev_seq if idx >= 0])
    bone_idx_set = sorted(set(bone_idx))
    bone_idx = [np.where(bone_idx == idx)[0] for idx in bone_idx_set]

    part_bone_disparity = [part_bone_disparity[bone_idx[i]].max(axis=0) for i in range(len(bone_idx))]

    part_bone_disparity = np.array(part_bone_disparity).reshape(-1, size, size)

    # draw keypoit
    key = joint_pos_image[0, :2].transpose(1, 0)  # num_joint x 2
    key = key[hpp.valid_keypoints]
    keypoint_mask = np.zeros((len(key), size, size))
    for i, (x, y) in enumerate(key):
        try:
            left = np.ceil(x - thickness).astype("int")
            right = np.ceil(x + thickness).astype("int")
            top = np.ceil(y - thickness).astype("int")
            bottom = np.ceil(y + thickness).astype("int")
            keypoint_mask[i, top:bottom, left:right] = (bottom >= 0) * (right >= 0)
        except:
            import pdb
            pdb.set_trace()
    return (camera_disparity.astype("float32"), mask.astype("float32"),
            part_bone_disparity.astype("float32"), keypoint_mask.astype("float32"))
