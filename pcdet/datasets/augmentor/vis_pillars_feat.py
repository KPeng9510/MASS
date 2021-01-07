import numpy as np
from PIL import Image
import torch

def show_feat(feat):
    img = (20 * feat).astype(np.uint8)
    feat_toshow = Image.fromarray(img)
    feat_toshow.show()

file_name = '0.npy'
gt_file = '../../../tools/gt_seg' + file_name
pillar_file = '../../../tools/pillar' + file_name
obser_file = '../../../tools/obser' + file_name

voxel_feat = np.load(gt_file)
print('Seg gt Shape: ', voxel_feat.shape)
print('Seg max: ', voxel_feat.max())

show_feat(voxel_feat[0, 0, :, :])

pillar_feat = np.load(pillar_file)
pillar_feat = np.clip(pillar_feat, 0, 1)
print('Pillar Shape: ', pillar_feat.shape)
pillar_slice_feat = pillar_feat[0, 1, :, :]
show_feat(pillar_slice_feat*12)

obser_feat = np.load(obser_file)*255
print('obser max: ', obser_feat.max())
show_feat(obser_feat[0, 0, :, :]*12)


