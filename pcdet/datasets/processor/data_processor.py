from functools import partial
from mapping import mapping
import numpy as np
import sys
from ...utils import box_utils, common_utils


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        #print(data_dict['points'].shape)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None, voxel_generator_2=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            voxel_generator_2 = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=60000
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator, voxel_generator_2 = voxel_generator_2)
        
        points = data_dict['points']
        points = data_dict['points_sp']
        indices = data_dict['indices']
        #print(points.shape)
        """
           add code for visibility
        """
        ori_points = points[:, [0,1,2,4]]
        #print(points[:,-1])
        voxel_size = self.voxel_size
        pc_range = self.point_cloud_range
        #print(pc_range)
        #print(self.voxel_size)
        origins = data_dict['origins']
        num_points = points.shape[0]
        num_original = num_points
        time_stamps = np.array([0],dtype=np.float32)
        time_stamps = points[indices[:-1], -1]  # counting on the fact we do not miss points from any intermediate time_stamps
        time_stamps = (time_stamps[:-1]+time_stamps[1:])/2
        time_stamps = [-1000.0] + time_stamps.tolist() + [1000.0]  # add boundaries
        time_stamps = np.array(time_stamps)
        num_original = indices[-1]
        #print(time_stamps)
        #print(points.shape)
        #sys.exit()
        if num_points > num_original:
            #print("this is test sample")
            original_points, sampled_points = ori_points[:num_original,:], ori_points[num_original:,:]
            visibility, original_mask, sampled_mask = mapping.compute_logodds_and_masks(
                original_points, sampled_points,origins,time_stamps,pc_range,min(voxel_size))
            points = np.concatenate((original_points[original_mask], sampled_points[sampled_mask]))
        else:
            
            visibility = mapping.compute_logodds(
                             ori_points, origins,time_stamps,pc_range,0.2)
        
        np.set_printoptions(threshold=sys.maxsize)
        visi_map = np.zeros([512, 512,3])
        visibility = np.int64(visibility)
        visibility = np.reshape(visibility,(40, 512,512))[0:40, :, :]
        visibility = np.transpose(visibility, (2,1,0))
        #print(visibility)
        #sys.exit()
        mask_occ = (visibility >= 1).nonzero()
        #print(mask_occ)
        mask_free = (visibility == 0).nonzero()
        mask_unknown = (visibility == -1).nonzero()
        visi_map[np.int64(mask_free[0]),np.int64(mask_free[1]),:] = np.array([255,0,0])/255
        visi_map[np.int64(mask_occ[0]),np.int64(mask_occ[1]), :] = np.array([0,255,0])/255
        visi_map[mask_unknown[0], mask_unknown[1], :] = np.array([0,0,255])/255
        #print(.shape)
        #visibility = np.pad(visibility, ((0,2),(0,0)), 'edge')
        data_dict['visibility'] = visibility
        #print(data_dict.keys())
        dense_points = data_dict['dense_point']
        points = data_dict['points'] 
        #print(points.shape)
        voxel_output = voxel_generator.generate(points)
        voxel_dense = voxel_generator.generate(dense_points)
        #print(voxel_dense[...,-1])
        #print(voxel_output['voxels'].shape)
        #sys.exit()
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['dense_pillar'] = voxel_dense['voxels']
        data_dict['dense_pillar_coords'] = voxel_dense['coordinates']
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)
        #print(2)
        #sys.exit()
        return data_dict
	
