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
file_path = "/mrtstorage/users/kpeng/nuscene_pcdet/data/nuscenes/v1.0-trainval/"
save_path = "/mrtstorage/users/kpeng/nu_lidar_seg/concat_lidar_flat_divided_new/"
def seg_concat():
    nusc = NuScenes(version='v1.0-trainval', dataroot='/mrtstorage/users/kpeng/nuscene_pcdet/data/nuscenes/v1.0-trainval/', verbose=True)
    for my_sample in nusc.sample:
        sample_data_token = my_sample['data']['LIDAR_TOP']
        ori_filename = nusc.get('sample_data', sample_data_token)['filename']
        #print(ori_filename)
        anno_data = torch.from_numpy(np.float32(np.fromfile("/mrtstorage/users/kpeng/nu_lidar_seg/processed_anno_new/"+sample_data_token+"_lidarseg.bin", dtype=np.uint8, count=-1$
        ori_data =  np.fromfile(file_path+ori_filename, dtype=np.float32, count=-1)
        ori_data = torch.from_numpy(ori_data.reshape(-1,5))
        des_data = torch.cat([ori_data, anno_data.unsqueeze(1)],dim=-1).flatten()

        des_data = des_data.numpy().tobytes()

        binfile = open(save_path+ori_filename, 'wb+')
        binfile.write(des_data)
        binfile.close()

    """
    file_path = "/mrtstorage/users/kpeng/nuscene_pcdet/data/nuscenes/v1.0-trainval/"
    pathDir =  os.listdir(file_path)
    list_semantic=[]
    dic = {}
    with open("/mrtstorage/users/kpeng/nu_lidar_seg/v1.0-trainval/lidarseg.json") as f:
        data = json.load(f)
    with open("/mrtstorage/users/kpeng/nuscene_pcdet/data/nuscenes/v1.0-trainval/v1.0-trainval/sample_data.json") as f:
        o_data = json.load(f)
        print(len(o_data))

    for frame in o_data:
        #dic["key"] = file_name
        data = np.fromfile(file_path+frame["filename"], dtype=np.float32, count=-1)
        dic.update({frame["token"]:data})
        #list_semantic.append(dic)
    #for frame in data: #lidarseg
    for sample_dict in  data:
        #child = os.path.join('%s/%s' % (filepath, file_name))
        #scene_name_split = file_name.split("__")[0]
        #print(scene_name_split)
        token = sample_dict["token"]
        anno_file_name = sample_dict["filename"]
        anno_data = torch.from_numpy(np.fromfile("/mrtstorage/users/kpeng/nu_lidar_seg/pos/"+anno_file_name, dtype=np.uint8, count=-1))

        orignal_data = torch.from_numpy(dic[token].reshape(-1,5))
        print(oroginal_data)
        end"""
if __name__ == "__main__":
    seg_concat()

