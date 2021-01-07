import numpy as np
import time
from collections import defaultdict
import cv2

from skimage import io
from PIL import Image


def show_feat(feat):
    img = (20 * feat).astype(np.uint8)
    feat_toshow = Image.fromarray(img)
    feat_toshow.show()


def get_rotation_scale_matrix2d(center, angle, scale):
    """
    :param center: [x, y]
    :param angle: angle in radians, >0: clockwise
    :param scale: >1 enlarge
    :return:
    """
    alpha = scale * np.cos(-angle) # minus only for this application
    beta = scale * np.sin(-angle)
    return np.array([[alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                     [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]])


def global_rotation_voxel_feat(voxel_feat, theta, scale=1.0):
    ny, nx, c = voxel_feat.shape  # H W C
    # rot_mat = get_rotation_scale_matrix2d((int(nx / 2), int(ny / 2)), theta, scale)
    rot_mat = cv2.getRotationMatrix2D((int(nx / 2), int(ny / 2)), -theta * 180 / np.pi, scale)
    voxel_feat = cv2.warpAffine(voxel_feat, rot_mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, 1)


def global_scale_voxel_feat(voxel_feat, scale, theta=0.0):
    ny, nx, c = voxel_feat.shape  # H W C
    # rot_mat1 = get_rotation_scale_matrix2d((nx//2, ny//2), theta, scale)
    rot_mat = cv2.getRotationMatrix2D((int(nx / 2), int(ny / 2)), -theta * 180 / np.pi, scale)
    voxel_feat = cv2.warpAffine(voxel_feat, rot_mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, 1)


def global_translate_voxel_feat(voxel_feat, dw, dh):
    ny, nx, c = voxel_feat.shape  # H W C
    mat = np.array([[1, 0, dw], [0, 1, dh]], dtype=np.float32)
    voxel_feat = cv2.warpAffine(voxel_feat, mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, 1)

gt_file = '~/000031.png'
gt = np.array(io.imread(gt_file), dtype=np.float32)
print(gt.shape)
show_feat(gt)
gt = np.expand_dims(gt, -1)
print(gt.shape)
gt_x = np.flipud(gt)
# show_feat(gt_x.squeeze(-1))
gt_y = np.fliplr(gt)
# show_feat(gt_y.squeeze())
gt_scale = global_scale_voxel_feat(gt, 1.2)
# show_feat(gt_scale.squeeze(-1))
gt_rot = global_rotation_voxel_feat(gt, 0.2)
# show_feat(gt_rot.squeeze(-1))
gt_trans = global_translate_voxel_feat(gt, 100, -50)
show_feat(gt_trans.squeeze(-1))
