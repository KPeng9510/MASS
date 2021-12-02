from collections import defaultdict
from pathlib import Path
import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#import torch.utils.data as torch_data
#from data_loader_odo import data_loader
#from ..utils import common_utils
#from .augmentor.data_augmentor import DataAugmentor
#from .processor.data_processor import DataProcessor
#from .processor.point_feature_encoder import PointFeatureEncoder
import sys
import mapping
#from voxelize import dense

def label_generator():
    file_name = "sample_nusc_all.pkl"
    save_path = "/cvhci/data/nuScenes/data"
    save_p = "/export/md0/dataset/MASS/visi_nusc_gaussian_noise/"
    open_file = open(file_name, "rb")
    files_seq = pickle.load(open_file)
    files_seq.reverse()
    open_file.close()
    locale_b = "/home/kpeng/pc14/kitti_odo/training/00/000377.bin"
    locate = "/home/kpeng/pc14/kitti_odo/training/04/000092.bin"
    #index = files_seq.index(locate)
    #index_2 = files_seq.index(locale_b)
    #index = np.int(len(files_seq)/2)
    for point_path in files_seq:
        #print(point_path)
        #sys.exit()
        #dense_path = save_path +str(point_path).split('/')[-2] +'/velodyne/'+ str(point_path).split('/')[-1]
        dense_path = point_path.split('/')[-1]
        points = np.fromfile(point_path, dtype=np.float32, count=-1).reshape(-1,5)[:,:4]
        #print(np.max(visi))
        #print(visi)
        #print(np.min(visi))
        #visi = np.transpose(visi,(1,2,0))[:,:,19:]
        #mask_1 = visi == 1 #occu
        #print(mask_1.sum())
        #mask_0 = visi == 0 #unknown
        #mask_n_1 = visi == -1
        #print(mask_n_1.sum())
        #visi += 1
        #visi[mask_n_1] = 20
        #visi[mask_0] = 0
        #visi+=1
        #visi[mask_1]=255 #occu
        #visi = np.clip(visi.sum(axis=-1), 0, 255)
        
        pc_range = np.array([-51.2, -51.2, -5, 51.2, 51.2, 3],dtype=np.float32)
        voxel_size = np.array([0.2,0.2,8],dtype=np.float32)
        #dense_gt[:,-1] = np.clip(dense_gt[:,-1],0,12)
        indices = np.cumsum([0] + [points.shape[0]]).astype(int)
        origins = np.array([[0,0,0]], dtype=np.float32)
        num_points = points.shape[0]
        num_original = num_points
        #time_stamps = np.array([-1000,0],dtype=np.float32)
        #time_stamps = np.zeros([num_points],dtype=np.float32)
        time_stamps = points[indices[:-1], -1]  # counting on the fact we do not miss points from any intermediate time_stamps
        time_stamps = (time_stamps[:-1]+time_stamps[1:])/2
        time_stamps = [-1000.0] + time_stamps.tolist() + [1000.0]  # add boundaries
        time_stamps = np.array(time_stamps)

        #indices =np.array([0],dtype=np.float32)
        visi = mapping.compute_visibility(points, origins,time_stamps,pc_range,0.2).reshape(512,512)
        #print(np.max(visi))
        #sys.exit()
        #visi = torch.from_numpy(visi).reshape(1*40*500*1000,1) # 2, 20, 500, 1000
        #print(dense_gt[:1,:])
        
        #sys.exit()
        #weight_5_class = [1,2,3,4]
        #weight_0_class = [0]
        #weight_1_class = [5,6,7,8,9,10,11,12]
        #dense_gt[:,weight_5_class]= dense_gt[:,weight_5_class]*5
        #dense_gt[:,weight_0_class]=dense_gt[:,weight_0_class]*0
        #dense_gt[:,weight_1_class]=dense_gt[:,weight_1_class]*1
        #for i in range(12):
        #    dense_gt[:,i]+=0.12-0.01*i
        #    #print(debse_gt)
        #visi = visi.flatten().tobytes()
        print(save_p+dense_path)
        save = save_p+dense_path.split('m')[0]+'.png'
        #f.write(visi)
        #f.close()
        print(np.max(visi))
        visi = np.clip(visi,0,255)
        visi = visi.astype(np.uint8)
        im =Image.fromarray(visi)
        im.save(save)
        #break
        sys.exit()
if __name__ == '__main__':
    label_generator()




