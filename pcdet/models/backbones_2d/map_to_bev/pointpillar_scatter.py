import torch
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
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        pillar_seg = batch_dict["pillar_seg_gt"]
        dense_seg = batch_dict["pillar_dense_gt"]
        dense_coor = batch_dict["dense_pillar_coords"]
        points_mean = batch_dict["points_mean"].squeeze()
        pillar_features = torch.cat([pillar_features, pillar_seg,points_mean],dim=-1)
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                68,
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

        """
           end
        """
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_dense = torch.stack(batch_spatial_dense, 0).view(batch_size, 1, 512,512)
        #print(batch_spatial_dense.size())
        #sys.exit()
        """
           merge
        """
        batch_spatial_features = batch_spatial_features.view(batch_size, (self.num_bev_features+4) * self.nz, self.ny, self.nx)
        batch_seg_labels = batch_spatial_features[:,-4,:,:].unsqueeze(1)
        
        zero_mask = batch_seg_labels == 0
        batch_seg_labels[zero_mask] = batch_spatial_dense[zero_mask]
        #print(batch_spatial_dense[zero_mask])
        #sys.exit()
        """
            end
        """
        #batch_spatial_features = batch_spatial_features.view(batch_size, (self.num_bev_features+4) * self.nz, self.ny, self.nx)
        #batch_seg_labels = batch_spatial_features[:,-4,:,:].unsqueeze(1)
        batch_dict["labels_seg"] = batch_seg_labels
        onehot_labels = one_hot(batch_seg_labels.to(torch.int64),16)
        batch_pointsmean = batch_spatial_features[:,-3:,:,:]
        batch_dict["pointsmean"] = batch_pointsmean
        
        batch_spatial_features = batch_spatial_features[:, :self.num_bev_features,:,:]
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['one_hot']=onehot_labels
        return batch_dict
