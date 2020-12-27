import os
import torch
import numpy as np
import scipy.misc as m
import functools
import operator
from torch.utils import data

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot,dirs,filename)
        for looproot,dirs, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
class data_loader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        augmentations=None,
        img_norm=True,
        version="cityscapes",
        test_mode=False,
        sequence_num=1,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.img_size = img_size
        
        
        self.files_seq = []
        for index in sequence_num:
            self.files_seq.append(recursive_glob(rootdir=os.path.join(self.root, "stuttgart_0"+str(index)), suffix=".png"))

        self.files = functools.reduce(operator.iconcat, self.files_seq, [])

        #if split == "train":
        #    sequence = ["00","01","02","03","04","05","06","07","09","10"]
        #else:
        #    sequence = ["08"]
        #file_list = []
        #for idx, scene in enumerate(sequence):
        #    file_list.extend(recursive_glob(self.root+scene+'/', suffix=".bin"))
    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        point_path = self.files[index].rstrip()
        points = m.imread(img_path)
        points = np.array(img, dtype=np.uint8)
        data_dict["points"] = 
        return img

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)

        img = torch.from_numpy(img).float()

        return img



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/home/kunyu/demoVideo/"
    dst = data_loader(local_path, is_transform=True, sequence_num=[1,2,3])
    bs = 1
    trainloader = data.DataLoader(dst, batch_size=1, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs = data_samples
        import pdb
        #pdb.set_trace()
        print(imgs.size())
