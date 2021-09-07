from collections import defaultdict
from pathlib import Path
from PIL import Image
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
#import torch.utils.data as torch_data
#from data_loader_odo import data_loader
#from ..utils import common_utils
#from .augmentor.data_augmentor import DataAugmentor
#from .processor.data_processor import DataProcessor
#from .processor.point_feature_encoder import PointFeatureEncoder
import sys
#from mapping import mapping
#from voxelize.voxelize import dense
color_map_1={
  "0" : [255, 255, 255],
  "1": [245, 150, 100],
  "4": [250, 80, 100],
  "16": [255, 0, 0],
  "7": [180, 30, 80],
  "3": [255, 0, 0],
  "9": [30, 30, 255],
  "8": [200, 40, 255],
  "2": [90, 30, 150],
  "11": [255, 0, 255],
  "12": [75, 0, 75],
  "14": [255, 150, 255],
  "15": [0, 255, 0],
  "5": [0, 60, 135],
  "13": [0, 255, 0],
  "10" :[0, 0, 255],
  "6": [255, 255, 50],
  "17": [80, 240, 150],
  "18": [150,240,255],
  "19": [0,255,255]
}
color_map ={
"0": [255, 255, 255],
"1": [112, 128, 144],
"2": [220, 20, 60],
"3": [255, 127, 80],
"4": [255, 158, 0],
"5": [233, 150, 70],
"6": [255, 61, 99],
"7": [0, 0, 230],
"8": [47, 79, 79],
"9": [255, 140, 0],
"10": [255, 99, 71],
"11": [0, 207, 191],
"12": [175, 0, 75],
"13": [75, 0, 75],
"14": [112, 180, 60],
"15": [222, 184, 135],
"16": [0, 175, 0]
}

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def colorized_image_generator():
    #file_name = "/home/kpeng/pc14/sample_test.pkl"
    save_path = "/home/kpeng/pc14/nuscenes/colorized_gt/"
    #open_file = open(file_name, "rb")
    file_path = "/home/kpeng/pc14/nuscenes/label_image_dense/LIDAR_TOP/"
    files_seq = recursive_glob(rootdir=file_path, suffix=".png")
    #files_seq = pickle.load(open_file)
    #open_file.close()
    #print(files_seq)
    #sys.exit()
    #locate = "/home/kpeng/pc14/kitti_odo/training/08/0000168.bin"
    #index = files_seq.index(locate)
    for point_path in files_seq:
        #dense_path = '/home/kpeng/pc14/nuscenes/label_image_dense/LIDAR_TOP/n015-2018-09-25-13-17-43+0800__LIDAR_TOP__1537852966648600.png'
        pointcloud = np.array(Image.open(point_path)).reshape(512,512,1) #np.fromfile(str(dense_path), dtype=np.float32, count=-1).reshape([512,512,1])
        print(np.sum(pointcloud==4))
        picture = np.zeros([512,512,3])
        #print(np.max(pointcloud))
        for i in range(0,17):
            mask = pointcloud[:,:,-1] == i
            #print(mask.shape)
            if mask.sum() == 0:
                continue
            picture[mask]=np.array(color_map[str(i)])/255
            #plt.figure()
            #picture=np.transpose()
        img_path = save_path +str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1].split('.')[0]+'.png'
        print(img_path)
        plt.imsave(img_path,picture)
        #sys.exit()
def label_generator():
    #file_name = "/home/kpeng/pc14/sample_test.pkl"
    save_path = "/mrtstorage/users/kpeng/nu_lidar_seg/label_image_dense/"
    #open_file = open(file_name, "rb")
    file_path = "/mrtstorage/users/kpeng/nu_lidar_seg/concat_lidar_flat_divided/new_2/samples/LIDAR_TOP/"
    #files_seq = pickle.load(open_file)
    files_seq = recursive_glob(rootdir=file_path, suffix=".bin")
    #open_file.close()
    #locate = "/home/kpeng/pc14/kitti_odo/training/10/000362.bin"
    #index = files_seq.index(locate)
    for point_path in files_seq:
        dense_path = point_path
        dense_gt = np.fromfile(str(dense_path), dtype=np.float32, count=-1).reshape([-1, 6])[:, :6]
        #print(np.max(dense_gt[:,-1]))
        pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],dtype=np.float32)
        voxel_size = np.array([0.2,0.2,8],dtype=np.float32)
        dense_gt[:,-1] = np.clip(dense_gt[:,-1],0,17)
        dense_gt = dense.compute_dense_gt(dense_gt, pc_range,voxel_size,17).reshape(1,17,512,512)
        dense_gt = torch.from_numpy(dense_gt).permute(0,2,3,1).reshape(1*512*512,17) # 2, 20, 500, 1000
        #print(torch.sum(torch.max(dense_gt[:,-1])))

        #sys.exit()
        weight_5_class = [2,3,4,5,6,7,10]
        weight_0_class = [0]
        weight_1_class = [1,8,9,11,12,13,14,15,16]
        dense_gt[:,weight_5_class]= dense_gt[:,weight_5_class]*5
        dense_gt[:,weight_0_class]=dense_gt[:,weight_0_class]*0
        dense_gt[:,weight_1_class]=dense_gt[:,weight_1_class]*1
        for i in range(17):
            dense_gt[:,i]+=0.16-0.01*i
            #print(debse_gt)
        #print(torch.sum(torch.max(dense_gt[:,16])))
        dense_gt = torch.argmax(dense_gt,dim=-1).view(512,512).numpy()
        #print(np.max(dense_gt))
        #sys.exit()
        #print(save_path+str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1])
        save=save_path+str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1].split('.')[0]+'.png'
        #sys.exit()
        dense_gt=np.clip(dense_gt,0,255).astype(np.uint8)
        im = Image.fromarray(dense_gt)
        print(save)
        im.save(save)
        #f.write(dense_gt)
        #f.close()

if __name__ == '__main__':
    #label_generator()
    colorized_image_generator()

