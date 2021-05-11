from collections import defaultdict
from pathlib import Path
import os
import pickle
import numpy as np
import torch.utils.data as torch_data
# from data_loader_odo import data_loader
from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
import sys
from pathlib import Path
from skimage import io
import time


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.root = Path("/home/kpeng/pc14/")
        self.gt_dense_bin_root = self.root / 'kitti_odo/training'
        self.gt_dense_img_root = self.root / 'dense_label'
        self.gt_obser_img_root = self.root / 'occupancy'
        self.gt_sparse_img_root= self.root / 'sparse_label'
        if training == True:
            self.split = "train"
            sequence = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
            file_name = "/home/kpeng/pc14/sample_test.pkl"
        else:
            self.split = "test"
            sequence = ["08"]
            file_name = "/home/kpeng/pc14/sample.pkl"

        self.files_seq = []
        # comment it out when generate sample file
        # for idx, scene in enumerate(sequence):
        #     self.files_seq.extend(recursive_glob(str(self.root) + '/sequences/' + scene + '/', suffix=".bin"))
        # print(self.files_seq)
        # open_file = open(file_name, "wb")
        # pickle.dump(self.files_seq, open_file)
        # open_file.close()
        # sys.exit()
        # end

        open_file = open(file_name, "rb")
        self.files_seq = pickle.load(open_file)
        open_file.close()
        # print(len(self.files_seq)

    @property
    def mode(self):

        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def get_dense_gt_bin(self, seq, index):
        f_file = self.gt_dense_bin_root / seq / ('%s.bin' % index)
        assert f_file.exists()
        return np.fromfile(str(f_file), dtype=np.float32, count=-1).reshape(500, 1000, 1)

    def get_dense_gt_img(self, seq, index):
        f_file = self.gt_dense_img_root / seq / ('%s.png' % index)
        assert f_file.exists()
        return np.array(io.imread(f_file), dtype=np.float32).reshape(500, 1000, 1)

    def get_grid_dense_gt_img(self, seq, index):
        f_file = self.gt_obser_img_root / seq / ('%015d.png' % int(index))
        assert f_file.exists()
        return np.array(io.imread(f_file), dtype=np.float32).reshape(500, 1000, 1)

    def get_sparse_gt_img(self, seq, index):
        f_file = self.gt_sparse_img_root / seq / ('%06d.png' % int(index))
        #print(f_file)
        assert f_file.exists()
        return np.array(io.imread(f_file), dtype=np.float32).reshape(500, 1000, 1)

    def get_obser_img(self, seq, index):
        f_file = self.gt_obser_img_root / seq / ('%06d.png' % int(index))
        assert f_file.exists()
        return np.array(io.imread(f_file), dtype=np.float32).reshape(500, 1000, 1)

    def __len__(self):
        return len(self.files_seq)

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        #t1=time.clock()
        data_dict = {}
        point_path = self.files_seq[index].rstrip()
        #if not point_path.exists():
        #    print(str(point_path))
        # print(point_path)
        point_path = "/home/kpeng/pc14/kitti_odo/training/"+point_path.split('/')[-2]+"/velodyne/"+point_path.split('/')[-1]
        points = np.fromfile(str(point_path), dtype=np.float32, count=-1).reshape([-1, 4])
        mask = common_utils.mask_points_by_range(points, self.point_cloud_range)
        points = points[mask]

        points = np.concatenate([points, np.zeros([points.shape[0], 1])], axis=-1)
        data_dict["points"] = points

        seq = str(point_path).split('/')[-3]
        idx = Path(point_path).stem

        data_dict['frame_id'] = seq+idx

        # TODO dense gt
        # seg_gt = self.get_dense_gt_bin(seq, idx)
        seg_gt = self.get_dense_gt_img(seq, idx)

        # get grid dense gt
        # seg_gt = self.get_grid_dense_gt_img(seq, idx)[:500, :1000, :]
        # remapping from franks if to pengs
        # seg_gt = seg_gt + 1
        # bkgd_mask = seg_gt == 31
        # seg_gt[bkgd_mask] = 0

        # sparse gt
        # seg_gt = self.get_sparse_gt_img(seq, idx)[:500, :1000, :]
        # remapping from franks if to pengs
        # seg_gt = seg_gt + 1
        # bkgd_mask = seg_gt == 31
        # seg_gt[bkgd_mask] = 0

        data_dict['labels_seg'] = seg_gt


        # load observations
        obser = self.get_obser_img(seq, idx)[:500, :1000, :]
        # print(obser.shape)
        data_dict['observations'] = obser

        data_dict = self.prepare_data(data_dict)

        seg_gt = np.transpose(data_dict['labels_seg'], (2, 0, 1))  # 1, 500, 1000
        data_dict['labels_seg'] = seg_gt

        obser = np.transpose(data_dict['observations'], (2, 0, 1))
        data_dict['observations'] = obser / 255  # normalization

        data_dict.pop('points', None)
        #print(t1-time.clock())
        #sys.exit()
        return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            data_dict = self.data_augmentor.forward(data_dict=data_dict)

        # debug data augmentation
        # gt_seg = data_dict['labels_gt']


        # data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['voxel_coords']) == 0:
            new_index = np.random.randint(self.__len__())
            print('Error frame id: ', data_dict['frame_id'])
            return self.__getitem__(new_index)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        # print("test")
        # sys.exit()
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        # data_dict.pop('points_sp', None)
        data_dict.pop('indices', None)
        data_dict.pop('origins', None)
        # print(data_dict.keys())
        # sys.exit()
        ret['frame_id'] = []
        for key, val in data_dict.items():
            # print('collate: ', key)
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'dense_point']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key == 'frame_id':
                    ret[key].append(data_dict['frame_id'])
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        # print('vis' in ret.keys())
        ret['batch_size'] = batch_size

        return ret
