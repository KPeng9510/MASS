#%matplotlib inline

from nuscenes import NuScenes
import os
import numpy as np
import torch
import json
import sys
import glob
import logging

logging.basicConfig(level=logging.DEBUG)
file_path = "/cvhci/data/nuscenes/" # path to the raw lidar data of key frame for nuscenes
save_path = "/cvhci/temp/kpeng/concat_lidar_flat_divided_new/" # path to the point cloud concatenated with semantic labels
if not os.path.exists(save_path+'samples/'):
    os.mkdir(save_path+'samples/')
    os.mkdir(save_path+'samples/'+'LIDAR_TOP')


def seg_concat():
    nusc = NuScenes(version='v1.0-trainval', dataroot='/cvhci/data/nuscenes/', verbose=True) #path to v1.0 trainval nuscenece meta data
    for my_sample in nusc.sample:
        sample_data_token = my_sample['data']['LIDAR_TOP']
        ori_filename = nusc.get('sample_data', sample_data_token)['filename']
        #/mrtstorage/users/kpeng/nu_lidar_seg/processed_anno_new/ is the path to the processed label generated through lidarseg_annotools
        anno_data = torch.from_numpy(np.float32(np.fromfile("/cvhci/temp/kpeng/processed_anno_new/"+sample_data_token.split('/')[-1]+"_lidarseg.bin", dtype=np.uint8, count=-1)))
        ori_data =  np.fromfile(file_path+ori_filename, dtype=np.float32, count=-1)
        ori_data = torch.from_numpy(ori_data.reshape(-1,5))
        des_data = torch.cat([ori_data, anno_data.unsqueeze(1)],dim=-1).flatten()

        des_data = des_data.numpy().tobytes()

        binfile = open(save_path+ori_filename, 'wb+')
        binfile.write(des_data)
        binfile.close()

if __name__ == "__main__":
    seg_concat()

