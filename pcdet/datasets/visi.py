from collections import defaultdict
from pathlib import Path
import os
import pickle
import torch
import numpy as np
# import torch.utils.data as torch_data
# from data_loader_odo import data_loader
# from ..utils import common_utils
# from .augmentor.data_augmentor import DataAugmentor
# from .processor.data_processor import DataProcessor
# from .processor.point_feature_encoder import PointFeatureEncoder
import sys
from mapping import mapping
# from voxelize import dense
from pathlib import Path


def label_generator():
    file_name = "/home/ki/input/kitti/semantickitti/dataset/sample.pkl"
    # save_path = Path("/home/ki/input/kitti/semantickitti/dataset/pillarseg/visibility")
    save_path = Path("/home/ki/hdd0/input/kitti/semantickitti/pillarseg/visibility")
    save_path.mkdir(parents=True, exist_ok=True)
    open_file = open(file_name, "rb")
    files_seq = pickle.load(open_file)
    open_file.close()
    # locate = "/home/kpeng/pc14/kitti_odo/training/09/000577.bin"
    # index = files_seq.index(locate)
    for point_path in files_seq:
        # dense_path = '/home/kpeng/training/' +str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1]
        points = np.fromfile(str(point_path), dtype=np.float32, count=-1).reshape([-1, 4])[:, :4]
        pc_range = np.array([-50.0, -25.0, -2.5, 50.0, 25.0, 1.5], dtype=np.float32)
        # voxel_size = np.array([0.1, 0.1, 8], dtype=np.float32)
        # dense_gt[:,-1] = np.clip(dense_gt[:,-1],0,12)
        indices = np.cumsum([0] + [points.shape[0]]).astype(int)
        origins = np.array([[0, 0, 0]], dtype=np.float32)
        num_points = points.shape[0]
        num_original = num_points
        # time_stamps = np.array([-1000,0],dtype=np.float32)
        time_stamps = np.zeros([num_points], dtype=np.float32)
        time_stamps = points[
            indices[:-1], -1]  # counting on the fact we do not miss points from any intermediate time_stamps
        time_stamps = (time_stamps[:-1] + time_stamps[1:]) / 2
        time_stamps = [-1000.0] + time_stamps.tolist() + [1000.0]  # add boundaries
        time_stamps = np.array(time_stamps)

        indices = np.array([0], dtype=np.float32)
        visi = mapping.compute_logodds(points, origins, time_stamps, pc_range, 0.1).reshape(1, 40, 500, 1000)
        # print((visi!=0).sum())
        # sys.exit()
        visi = torch.from_numpy(visi).reshape(1 * 40 * 500 * 1000, 1)  # 2, 20, 500, 1000
        # print(dense_gt[:1,:])

        # sys.exit()
        # weight_5_class = [1,2,3,4]
        # weight_0_class = [0]
        # weight_1_class = [5,6,7,8,9,10,11,12]
        # dense_gt[:,weight_5_class]= dense_gt[:,weight_5_class]*5
        # dense_gt[:,weight_0_class]=dense_gt[:,weight_0_class]*0
        # dense_gt[:,weight_1_class]=dense_gt[:,weight_1_class]*1
        # for i in range(12):
        #    dense_gt[:,i]+=0.12-0.01*i
        #    #print(debse_gt)
        visi = visi.flatten().numpy().astype(np.float32).tobytes()
        f_path = save_path / (str(point_path).split('/')[-3] + '/' + str(point_path).split('/')[-1])
        f_path.parent.mkdir(parents=True, exist_ok=True)
        print(f_path)
        f = open(str(f_path), 'wb')
        f.write(visi)
        f.close()


if __name__ == '__main__':
    label_generator()
