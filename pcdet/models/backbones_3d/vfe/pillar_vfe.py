import torch

from ....ops.roiaware_pool3d import roiaware_pool3d_utils
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

import sys
from torch_geometric.nn import FeaStConv
from knn_cuda import KNN
from torch_cluster import fps
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
class Graph_attention(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Graph_attention, self).__init__()
        #self.Linear1 = nn.Sequential(nn.Linear(2*in_channels, hidden_channels),nn.ReLU())
        self.feast1=FeaStConv(in_channels, in_channels)
        self.feast1_2=FeaStConv(in_channels, hidden_channels)
        self.relu=nn.ReLU()
        #self.Linear2 = nn.Sequential(nn.Linear(2*hidden_channels,in_channels),nn.ReLU())
        self.feast2=FeaStConv(hidden_channels,in_channels)
        self.feast2_2=FeaStConv(in_channels, in_channels)
        self.fc1 = nn.Linear(2*in_channels, in_channels)
        self.sigmoid = nn.Softmax(dim=0)
        #self.PillarScatter(denseshape, num_input_features=9)
        self.batch_size=2
    def forward(self,x):
        b,p,c = x.size()
        #print(x.size())
        #xyz_x = x[:,:,:3]
        #x = self.PillarScatter(x,coors,self.batch_size)
        
        
        #features = x[batch,:,:,:]
        #print(x.size())
        
        #y = (x[:, :, :].sum(dim=1, keepdim=True) / num_voxels.type_as(x).view(-1,1,1))
        y = torch.max(x, dim=1, keepdim=True)[0].view(b,c)
        voxel_number = y.size()[0]
        
        #row = torch.arange(0,b).unsqueeze(-1).repeat(1,b).view(1,b*b)
        
        batch = torch.zeros(b).long().cuda()
        col = fps(y, batch, ratio=0.02, random_start=True).cuda()
        fps_p = len(col)
        row = torch.arange(0,b).unsqueeze(-1).repeat(1,fps_p).view(1,b*fps_p).long().cuda()
        col = col.repeat(b,1).view(1,b*fps_p)
        edge_index = torch.cat([row,col], dim=0)
        #print(edge_index.size())
        #k=30
        #knn = KNN(k=30, transpose_mode=True).cuda()

        #dist, col = knn(xyz[:,:,:3], xyz[:,:,:3]) # 1, 6000, 20
        """
        row = torch.arange(start=0, end=voxel_number).unsqueeze(-1).repeat(1,k).resize(voxel_number*k).long().cuda()
        col = col.view(-1)
        mask = 1 - torch.isinf(dist).view(-1).int()
        row, col = row.view(-1)[mask.long()], col.view(-1)[mask.long()]
        edge_index = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)"""
        #print(y.size(), edge_index.size())
        #y=y.squeeze()
        ##y = torch.cat([xyz,y], dim=-1)
        out= self.relu(self.feast1(y,edge_index))
        #out = out
        #print(out.size())
        ##out = torch.cat([xyz, out],dim=-1)
        out = self.relu(self.feast1_2(out,edge_index))
        ##out = torch.cat([xyz, out], dim=-1)
        out = self.relu(self.feast2(out, edge_index))
        ##out = torch.cat([xyz, out], dim=-1)
        attention=self.sigmoid(self.feast2_2(out,edge_index))
        #print(attention.size())
        #attention = torch.stack(attention)
        out = x*attention.unsqueeze(1)
        #print(out.size())
        out=torch.cat([out,x], dim=-1)
        #print(out.size())
        out=self.fc1(out)
        return out
def PCA(X, k=2):
    
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    # SVD
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

class lstm_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_attention,self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,1,batch_first=True,bidirectional=True)
        self.reduce = nn.Linear(4*input_dim,input_dim)
        
        
        #self.relu = nn.ReLU()
        self.softmax = nn.Sigmoid()
    def forward(self,features, points_mean):
        p,_,c = features.size()
        X_1D = PCA(points_mean.squeeze(), 1)
        #print(points_mean.size())
        #X_1D = lpp.fit_transform(points_mean.squeeze())

        #X_1D = torch.from_numpy(gpumap.GPUMAP(n_neighbors=5, n_components=1).fit_transform(points_mean.squeeze().cpu().numpy())).cuda()
        #r,phi,theta = cart2sph(points_mean)
        #_,indices = torch.sort(torch.square(r.squeeze())*torch.square(phi.squeeze()))
        #print(X_1D.size())
        _,indices = torch.sort(X_1D.squeeze())
        indices = indices.squeeze()
        features = features[indices,:,:].squeeze()
        #print(features.size())
        feature_max = torch.max(features, dim=1, keepdim=True)[0]
        feature_mean = torch.mean(features,dim=1,keepdim=True)
        #center_max = torch.mean(feature_max, dim=0, keepdim=True).repeat(p,1,1)
        #center_mean = torch.mean(feature_mean,dim=0,keepdim=True).repeat(p,1,1)

        #distance_max = torch.sqrt(torch.sum(torch.square(torch.abs(feature_max - center_max)),dim=-1,keepdim=True)).squeeze()
        #distance_mean = torch.sqrt(torch.sum(torch.square(torch.abs(feature_mean - center_mean)),dim=-1,keepdim=True)).squeeze() #p,1,1

        #_, indices_max = torch.sort(distance_max,0,descending=False)
        #_, indices_mean = torch.sort(distance_mean,0,descending=False)
        
        #sorted_max = feature_max[indices_max,:,:]
        #sorted_mean = feature_mean[indices_mean,:,:]

        out_max = self.lstm(feature_max)[0]
        out_mean = self.lstm(feature_mean)[0]

        #canvas_max = torch.zeros([p,1,2*c], dtype=feature_max.dtype,device=feature_max.device)
        #canvas_mean = torch.zeros([p,1,2*c], dtype=feature_mean.dtype,device=feature_mean.device)

        #canvas_max[indices_max,:,:] = out_max
        #canvas_mean[indices_mean,:,:] = out_mean
        
        final = self.softmax(self.reduce(torch.cat([out_max,out_mean], dim=-1)))
        canvas = torch.zeros([p,1,c],dtype=out_max.dtype,device=out_max.device)
        canvas[indices,:,:]=final
        return canvas
class pointsmean_attention(nn.Module):
    def __init__(self, c_num, p_num):
        super(pointsmean_attention, self).__init__()
        self.fc1=nn.Sequential(
        nn.Linear(c_num+3, 1),
        nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
        nn.Linear(20, 1),
        nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, voxel_center, feature):

        voxel_center_repeat = voxel_center.repeat(1, feature.shape[1],1)
        voxel_feat_concat = torch.cat([feature,voxel_center_repeat], dim=-1)
        feat_2 = self.fc1(voxel_feat_concat)
        feat_2 = feat_2.permute(0,2,1).contiguous()
        voxel_feat_concat = self.fc2(feat_2)
        voxel_attention_weight = self.sigmoid(voxel_feat_concat)
        return voxel_attention_weight

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
        self.relu = nn.ReLU()
        self.lstm_attention = lstm_attention(num_point_features, num_point_features)
        self.graph_attention=Graph_attention(num_point_features, 4)
        #self.graph_attention_2=Graph_attention(num_input_features,4)
        self.pointsmean_attention = pointsmean_attention(num_point_features, 100)
        #self.pointsmean_attention_2 = pointsmean_attention(num_input_features, 100)
        #self.feastconv = FeaStConv(20, num_input_features-1, 3, True, 1)
        self.FC1=nn.Sequential(
        nn.Linear(2*num_point_features, num_point_features),
        nn.ReLU(inplace=True),
        )
        self.FC2=nn.Sequential(
        nn.Linear(num_point_features,num_point_features),
            nn.ReLU(inplace=True),
        )


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
        coords = batch_dict['voxel_coords']
        batch_size = coords[:, 0].max().int().item() + 1
        # dense_gt = batch_dict['dense_gt'].permute(0,2,3,1).reshape(batch_size*500*1000,13) # 2, 20, 500, 1000
        #print(dense_gt[:1,:])

        #sys.exit()
        # weight_5_class = [1,2,3,4]
        # weight_0_class = [0]
        # weight_1_class = [5,6,7,8,9,10,11,12]
        # dense_gt[:,weight_5_class]= dense_gt[:,weight_5_class]*5
        # dense_gt[:,weight_0_class]=dense_gt[:,weight_0_class]*0
        # dense_gt[:,weight_1_class]=dense_gt[:,weight_1_class]*1
        # for i in range(12):
        #     dense_gt[:,i]+=0.12-0.01*i
        # #print(debse_gt)
        # dense_gt = torch.argmax(dense_gt,dim=-1)
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
        # v,p,c = voxel_features.size()
        
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
        # batch_dict["pillar_dense_gt"] = dense_gt
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
        #coor = torch.zeros([3,512,512], dtype=f_center.dtype, device=f_center.device)
        #x = torch.linspace(0,512,512) #*self.voxel_x + self.x_offset
        #z = torch.linspace(0,1,1)
        #y = torch.linspace(0,512,512)
        #grid_x,grid_y,grid_z = torch.meshgrid(x,y,z)
        #coor = torch.cat([(grid_x*self.voxel_x + self.x_offset).unsqueeze(-1), (grid_y*self.voxel_y + self.y_offset).unsqueeze(-1), (grid_z*self.voxel_z + self.z_offset).unsqueeze(-1)], dim=-1)
        #coor = coor.view(512*512,3)
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
        # batch_dict["points_mean"]=center

        # batch_dict["points_coor"]=coor
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        batch_spatial_features={}
        container = torch.zeros_like(features)
        for index in range(batch_size):
            
            batch_mask = coords[:, 0] ==index
            batch_features = features[batch_mask, :]
            #batch_spatial_features.append(spatial_feature)
            batch_points_mean = points_mean[batch_mask,:]
            lstm_attention = self.lstm_attention(batch_features,batch_points_mean)
            
            batch_features = lstm_attention*batch_features
            features_ori = batch_features
            batch_features = self.graph_attention(batch_features)
            
            voxel_attention1 = self.pointsmean_attention(batch_points_mean, batch_features)
            batch_features = voxel_attention1 * batch_features
            out1 = torch.cat([batch_features, features_ori],dim=-1)
            #print(out1.size())
            #point_gcn_feature = self.gcn(inputPC=features).cuda()
            out1 = self.FC1(out1)
            #features = self.graph_attention(features, num_voxels)
            #features = self.graph_attention_2(out1)
            #voxel_attention2 = self.pointsmean_attention_2(points_mean, features)
            #features = voxel_attention2 * features
            #out2 = out1 + features
            out1 = self.FC2(out1)
            container[batch_mask,:] = out1
            # Forward pass through PFNLayers
            #final_result.append(features)
            #len1,len2 = final_result[0].size()[0], final_result[1].size()[0]
            #features = torch.cat([final_result[0], final_result[1]],dim=0)
        """
           dense gt
        """
        features = container
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
