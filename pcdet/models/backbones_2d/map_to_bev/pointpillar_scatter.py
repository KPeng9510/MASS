import torch
import numpy as np
import torch.nn as nn
import sys


def one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        # self.nx = 1001

        self.conv_obser = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        # self.w_pillar = nn.Sequential(
        #     nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        # )

        # self.w_obser = nn.Sequential(
        #     nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        # )

        assert self.nz == 1
        # self.num = 0

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        # print(coords.shape)
        # pillar_seg = batch_dict["pillar_seg_gt"]
        batch_size = coords[:, 0].max().int().item() + 1
        # dense_seg = batch_dict["labels_seg"].resize(batch_size,1,500,1000)
        # dense_coor = batch_dict["dense_pillar_coords"]
        # print(pillar_features.dtype)
        # sys.exit()
        # visibility = batch_dict['vis'].to(torch.float32).contiguous().permute(0, 3, 1,
        #                                                                       2).contiguous()  # 2, 40, 512, 512
        batch_spatial_features = []
        # batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                64,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        """
           dense gt
        """
        '''
        batch_spatial_dense = []
        batch_size = dense_coor[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_dense = torch.zeros(
                1,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = dense_coor[:, 0] == batch_idx
            this_coords = dense_coor[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            gts = dense_seg[batch_mask, :]
            gts = gts.t()
            spatial_dense[:, indices] = gts
            #print(indices.size())
            #print(spatial_dense)
            batch_spatial_dense.append(spatial_dense)
            '''

        """
           end
        """
        # torch.autograd.set_detect_anomaly(True)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # batch_spatial_dense = torch.stack(batch_spatial_dense, 0).contiguous().view(batch_size, 1, self.ny,self.nx)
        # print(batch_spatial_dense.size())
        # sys.exit()
        """
           merge
        """
        # batch_spatial_features = batch_spatial_features.contiguous().view(batch_size, (self.num_bev_features+3) * self.nz, self.ny, self.nx)
        # batch_seg_labels = batch_spatial_features[:,-4,:,:].unsqueeze(1)

        # zero_mask = batch_seg_labels == 0
        # batch_seg_labels = dense_seg
        # print(batch_spatial_dense[zero_mask])
        # sys.exit()
        """
            end
        """
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                             self.nx)
        observations = batch_dict['observations'].contiguous().view(batch_size, 1, self.ny, self.nx)
        observations = self.conv_obser(observations)

        # Attentional fusion module
        # weight_pillar = self.w_pillar(batch_spatial_features)
        # weight_obser = self.w_obser(observations)
        # weight = torch.softmax(torch.cat([weight_pillar, weight_obser], dim=1), dim=1)
        # batch_spatial_features = batch_spatial_features * weight[:, 0:1, :, :] + observations * weight[:, 1:, :, :]

        # concat fusion module
        batch_spatial_features = torch.cat([batch_spatial_features, observations], dim=1)

        batch_dict['spatial_features'] = batch_spatial_features.contiguous()

        """
        # np.save('pillar' + str(self.num), batch_spatial_features[:, 0:2, :, :].cpu().detach().numpy())
        # batch_seg_labels = batch_spatial_features[:,-4,:,:].unsqueeze(1)
        # batch_dict['labels_seg'] = batch_seg_labels
        # line = torch.zeros([2,1,512,16]).cuda()
        # for i in range(16):
        #    line[:,:,:,-1]=i
        # batch_seg_labels = torch.cat([batch_seg_labels, line], dim=-1)
        # onehot_labels = one_hot(batch_seg_labels.to(torch.int64),16)
        # onehot_labels = onehot_labels[:,:,:,:512]
        # batch_pointsmean = batch_spatial_features[:,64:,:,:]
        # batch_dict['pointsmean'] = batch_pointsmean
        # torch.autograd.set_detect_anomaly(True)
        # batch_spatial_features = batch_spatial_features[:, :self.num_bev_features, :, :]
        # re_f = self.zp(batch_spatial_features)
        re_f = self.conv_pillar(batch_spatial_features)
        # visibility = self.zp(visibility)
        visibility = self.relu(self.conv_visi(visibility.permute(0, 1, 3, 2)))
        re_v = self.relu(self.conv_visi_2(visibility))
        # re_v = visibility
        # re_f = batch_spatial_features
        # print(re_v.dtype)
        # print(re_f.dtype)
        # sys.exit()
        attention = self.softmax(torch.cat([re_v, re_f], dim=1))
        att1 = attention[:, 0, :, :]
        att2 = attention[:, 1, :, :]
        re_v = att1.unsqueeze(1).repeat(1, 64, 1, 1).contiguous() * re_v
        re_f = att2.unsqueeze(1).repeat(1, 64, 1, 1).contiguous() * re_f
        batch_spatial_features = re_v + re_f
        batch_dict['spatial_features'] = batch_spatial_features
        # batch_dict['one_hot']=onehot_labels
        # print('vis' in batch_dict.keys())
        """

        # debug data augmentation
        # file_name = str(self.num)
        # self.num += 1
        # gt_seg_tosave = batch_dict['labels_seg'].cpu().numpy()
        # np.save('gt_seg' + file_name, gt_seg_tosave)
        #
        # obser = batch_dict['observations'].cpu().numpy()
        # np.save('obser' + file_name, obser)
        #
        # print('saving: ', self.num)

        return batch_dict
