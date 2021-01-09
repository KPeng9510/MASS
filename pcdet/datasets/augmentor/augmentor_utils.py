import numpy as np
import cv2

from ...utils import common_utils


def get_rotation_scale_matrix2d(center, angle, scale):
    """
    :param center: [x, y]
    :param angle: angle in radians, >0: clockwise
    :param scale: >1 enlarge
    :return:
    """
    alpha = scale * np.cos(-angle)
    beta = scale * np.sin(-angle)
    return np.array([[alpha, beta, center[0] - beta * center[1]],
                     [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]])


def global_rotation_voxel_feat(voxel_feat, theta, scale=1.0):
    ny, nx, c = voxel_feat.shape  # H W C
    # rot_mat = get_rotation_scale_matrix2d((int(nx / 2), int(ny / 2)), theta, scale)
    rot_mat = cv2.getRotationMatrix2D((int(nx / 2), int(ny / 2)), -theta * 180 / np.pi, scale)
    voxel_feat = cv2.warpAffine(voxel_feat, rot_mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, -1)


def global_scale_voxel_feat(voxel_feat, scale, theta=0.0):
    ny, nx, c = voxel_feat.shape  # H W C
    # rot_mat = get_rotation_scale_matrix2d((int(nx / 2), int(ny / 2)), theta, scale)
    rot_mat = cv2.getRotationMatrix2D((int(nx / 2), int(ny / 2)), -theta * 180 / np.pi, scale)
    voxel_feat = cv2.warpAffine(voxel_feat, rot_mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, -1)


def random_flip_along_x(gt_seg, points, observations):
    """
    Args:
        gt_seg: 500, 1000, 1
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 1] = -points[:, 1]
        gt_seg = np.flipud(gt_seg)
        if observations is not None:
            observations = np.flipud(observations)
    return gt_seg, points, observations


def random_flip_along_y(gt_seg, points, observations):
    """
    Args:
        gt_seg: 500, 1000, 1
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 0] = -points[:, 0]
        gt_seg = np.fliplr(gt_seg)
        if observations is not None:
            observations = np.fliplr(observations)
    return gt_seg, points, observations


def global_rotation(gt_seg, points, observations, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_seg = global_rotation_voxel_feat(gt_seg, noise_rotation)
    if observations is not None:
        observations = global_rotation_voxel_feat(observations, noise_rotation)
    return gt_seg, points, observations


def global_scaling(gt_seg, points, observatjions, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_seg, points, observatjions
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_seg = global_scale_voxel_feat(gt_seg, noise_scale)
    if observatjions is not None:
        observatjions = global_scale_voxel_feat(observatjions, noise_scale)
    return gt_seg, points, observatjions


def global_translate_voxel_feat(voxel_feat, dw, dh):
    ny, nx, c = voxel_feat.shape  # H W C
    mat = np.array([[1, 0, dw], [0, 1, dh]], dtype=np.float32)
    voxel_feat = cv2.warpAffine(voxel_feat, mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, -1)


def global_translate(gt_seg, points, observations, noise_translate_std):
    """
    Apply global translation to gt_boxes and points.
    """
    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])

    std_x, std_y, std_z = noise_translate_std
    noise_translate = np.array([np.random.normal(0, std_x, 1),
                                np.random.normal(0, std_y, 1),
                                np.random.normal(0, std_z, 1)]).T  # 1 3
    # clip to 3*std
    noise_translate = np.clip(noise_translate, [-3.0*std_x, -3.0*std_y, -3.0*std_z], [3.0*std_x, 3.0*std_y, 3.0*std_z])
    points[:, :3] += noise_translate

    dw = noise_translate[0, 0] // 0.1
    dh = noise_translate[0, 1] // 0.1
    gt_seg = global_translate_voxel_feat(gt_seg, dw, dh)
    if observations is not None:
        observations = global_translate_voxel_feat(observations, dw, dh)

    return gt_seg, points, observations
