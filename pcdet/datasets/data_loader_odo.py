import os
import torch
import numpy as np
import scipy.misc as m
import functools
import operator
from torch.utils import data
import sys
def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot,filename)
        for looproot,_, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
class data_loader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        version="kitti-odo",
        test_mode=False,
    ):
        self.root = "/home/ki/input/kitti/semantickitti/dataset/sequences/"
        
        self.split = split
        
        
        self.data_dict = {}
        self.files_seq = []
        #for index in sequence_num:
        #self.files_seq.extend(recursive_glob(rootdir=self.root, suffix=".bin"))
        #print(self.files_seq)
        #self.files = functools.reduce(operator.iconcat, self.files_seq, [])

        if split == "train":
            sequence = ["00","01","02","03","04","05","06","07","09","10"]
        else:
            sequence = ["08"]
        #file_list = []
        for idx, scene in enumerate(sequence):
            self.files_seq.extend(recursive_glob(self.root+scene+'/', suffix=".bin"))
        print(len(self.files_seq))
    def __len__(self):
        """__len__"""
        return len(self.files_seq)

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        point_path = self.files_seq[index].rstrip()
        points = np.fromfile(str(point_path), dtype=np.float32, count=-1).reshape([-1, 4])[:, :4]
        dense_path = '/home/ki/hdd0/input/kitti/semantickitti/dense/' +str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1]
        dense_gt = np.fromfile(str(dense_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :5]
        self.data_dict["dense_gt"]=dense_gt
        self.data_dict["points"] = points
        return self.data_dict



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])
    local_path = "/home/kunyu/pc14/kitti_odo/training/"
    dst = data_loader(local_path)
    bs = 1
    trainloader = data.DataLoader(dst, batch_size=1, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        pcs = data_samples
        #import pdb
        #pdb.set_trace()
        print(pcs['dense_gt'].shape)
        sys.exit()
