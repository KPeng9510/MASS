import torch

from ....ops.roiaware_pool3d import roiaware_pool3d_utils
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

import sys

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)
        num_point_features=4
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        #print(batch_dict.keys())
        #gt_boxes = batch_dict["gt_boxes"]
        #print(batch_dict["gt_names"].size())
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        #print(voxel_features.size())
        dense_gt = batch_dict['dense_gt'].permute(0,2,3,1).reshape(2*500*1000,13) # 2, 20, 500, 1000
        #print(dense_gt[:1,:])

        #sys.exit()
        weight_5_class = [1,2,3,4]
        weight_0_class = [0]
        weight_1_class = [5,6,7,8,9,10,11,12]
        dense_gt[:,weight_5_class]= dense_gt[:,weight_5_class]*5
        dense_gt[:,weight_0_class]=dense_gt[:,weight_0_class]*0
        dense_gt[:,weight_1_class]=dense_gt[:,weight_1_class]*1
        for i in range(12):
            dense_gt[:,i]+=0.12-0.01*i
        #print(debse_gt)
        dense_gt = torch.argmax(dense_gt,dim=-1)
        #print(dense_gt[:100])
        #print(dense_gt)
        #sys.exit()
        #coor = batch_dict['dense_pillar_coords']
        """
        merge sem gt
        """
        
        #mask_zero = voxel_features[:,:,-2] == 0
        #seg_gt = voxel_features[:,:,5]
        #seg_get[mask_zero] == dense[:,:,-2][mask_zero]
        """
        end
        """
        #print(batch_dict.keys())
        #sys.exit()
        v,p,c = voxel_features.size()
        
        """
        for i in range(gt_boxes.size()[0]):
            points = voxel_features.resize(v*p,c)
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points[:, 0:3].unsqueeze(dim=0).float().cuda(),
                gt_boxes[:, 0:7].unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0)
            torch.set_printoptions(profile="full")
            print(torch.max(box_idxs_of_pts))
            print(gt_boxes.size())
        """
        voxel_features= voxel_features[:,:,:4]

        """
        encode for segmentation gt for each pillar

        """
        """
        zero_mask = dense_gt == 0
        length = dense_gt[zero_mask].size()[0]
        dense_gt_min = torch.min(dense_gt, dim=1,keepdim=True)[0]
        noise = torch.linspace(-20,-length-20-1,length)
        dense_gt[zero_mask] = noise.cuda()
        dense_gt_after = torch.mode(dense_gt.squeeze(),dim=-1,keepdim=True)[0]
        torch.set_printoptions(profile="full")
        dense_gt_max = torch.max(dense_gt.squeeze(),dim=-1,keepdim=True)[0]
        mask_max = dense_gt_max <0
        dense_gt_max[mask_max]=0

        mask = dense_gt_after < dense_gt_min
        dense_gt_after[mask] = dense_gt_max[mask]
        """
        batch_dict["pillar_dense_gt"] = dense_gt
        """
        encode end for segmentation gt for each pillar

        """

        """
        encode for dense segmentation gt for each pillar

        """
        """
        zero_mask = seg_gt == 0
        length = seg_gt[zero_mask].size()[0]
        seg_gt_min = torch.min(seg_gt, dim=1,keepdim=True)[0]
        noise = torch.linspace(-20,-length-20-1,length)
        seg_gt[zero_mask] = noise.cuda()
        seg_gt_after = torch.mode(seg_gt.squeeze(),dim=-1,keepdim=True)[0]
        torch.set_printoptions(profile="full")
        seg_gt_max = torch.max(seg_gt.squeeze(),dim=-1,keepdim=True)[0]
        mask_max = seg_gt_max <0
        seg_gt_max[mask_max]=0

        mask = seg_gt_after < seg_gt_min
        seg_gt_after[mask] = seg_gt_max[mask]
        batch_dict["pillar_seg_gt"] = seg_gt_after
        """
        #print(seg_gt_after)
        #sys.exit()
        """
        encode end for segmentation gt for each pillar

        """

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:,: , :3])
        center = torch.zeros_like(voxel_features[:,1,:3]).view(voxel_features.size()[0],1,3)
        coor = torch.zeros([3,512,512], dtype=f_center.dtype, device=f_center.device)
        x = torch.linspace(0,512,512) #*self.voxel_x + self.x_offset
        z = torch.linspace(0,1,1)
        y = torch.linspace(0,512,512)
        grid_x,grid_y,grid_z = torch.meshgrid(x,y,z)
        coor = torch.cat([(grid_x*self.voxel_x + self.x_offset).unsqueeze(-1), (grid_y*self.voxel_y + self.y_offset).unsqueeze(-1), (grid_z*self.voxel_z + self.z_offset).unsqueeze(-1)], dim=-1)
        coor = coor.view(512*512,3)
        center[:,:,0] = (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        center[:,:,1] = (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        center[:,:,2] = (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        batch_dict["points_mean"]=center

        batch_dict["points_coor"]=coor
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


            
