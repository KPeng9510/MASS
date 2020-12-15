from .detector3d_template import Detector3DTemplate
from .segmentation_head import FCNMaskHead
import sys
from .erfnet import Net
import os
import torch.nn.functional as F
import torch
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
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
    one_hot = torch.cuda.FloatTensor(labels.size()[0], num_classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1) 
    return target

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.segmentation_head = FCNMaskHead()
    def forward(self, batch_dict):
        module_index = 0
        #print(batch_dict.keys())
        for cur_module in self.module_list:
            module_index += 1
            batch_dict = cur_module(batch_dict)
            #print(batch_dict.keys())
            #points_mean = batch_dict['pointsmean']
            #batch_size,h,w = points_mean.size()
            #print(batch_dict["gt_boxes"][0,:,-1])
            if module_index == 4:
                """
                  encode bbox
                """
                points_mean = batch_dict["pointsmean"]
                #print(points_mean.size())
                gt_boxes = batch_dict["gt_boxes"]
                batch,c,h,w = points_mean.size()
                dict_seg = []
                dict_cls_num = []
                
                
                for i in range(gt_boxes.size()[0]):
                    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
                    points = points_mean[i,:,:,:].resize(c,h*w).permute(1,0)
                    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points[:, 0:3].unsqueeze(dim=0).float().cuda(),
                    gt_boxes[i,:, 0:7].unsqueeze(dim=0).float().cuda()
                    ).long().squeeze(dim=0)
                    
                    gt_boxes_indx = gt_boxes[i,:,-1]
                    
                    """
                    if i == 1:
                        sys.exit()
                    """
                    #nonzero_number = torch.sum(nonzero_mask.float())
                    
                    #print(nonzero_number)
                    #print(gt_boxes_indx.size())
                    #if i == 1:
                    #    sys.exit()
                    #gt_boxes_indx = gt_boxes_indx[:nonzero_number.int()]
                    gt_boxes_indx = torch.cat([torch.Tensor([0]).cuda(),gt_boxes_indx],dim=0)
                    box_idxs_of_pts +=1
                    target_cr = gt_boxes_indx[box_idxs_of_pts.long()]
                    #print(target_cr)
                    #print(max(box_idxs_of_pts))
                    
                    #sys.exit()
                    #torch.set_printoptions(profile="full")
                    #if i == 1:
                    #    sys.exit()
                    #mask = (box_idxs_of_pts == -1) & (box_idxs_of_pts >= nonzero_number)
                    #box_idxs_of_pts[mask] = -1
                    #box_idxs_of_pts += 1
                    #print(target_cr)
                    limit = torch.max(target_cr)
                    
                    #sys.exit()
                    #print(limit)
                    #print(limit)
                    target_cr = target_cr.view(1,1,h,w)
                    target_label = torch.zeros([1,1,h,12], dtype=target_cr.dtype, device = target_cr.device)
                    for i in range(12):
                        target_label[:,:,:,i] = i
                    target_cr = torch.cat([target_cr,target_label],dim=-1)
                    box_idxs_pillar = one_hot(target_cr.to(torch.int64), 16)
                    box_idxs_pillar = box_idxs_pillar[:,:,:,:w]
                    #driveable_area = torch.zeros([1,1,h,w],dtype=box_idxs_pillar.dtype, device=box_idxs_pillar.device)
                    #box_idxs_pillar = torch.cat([box_idxs_pillar[:,:limit.int()-1,:,:],driveable_area,box_idxs_pillar[:,-1,:,:].unsqueeze(1)],dim=1)
                    """
                    for i in range(12):
                         print(target_cr.size())
                         #sys.exit()
                         judgement = (target_cr.view(1,1,w,h).int() == i)
                         print(judgement)
                         if ((target_cr == i).byte().float().nonzero().size()[0] == 0):
                             new_tensor = torch.zeros([1,1,h,w],dtype=box_idxs_pillar.dtype, device=box_idxs_pillar.device)
                             box_idxs_pillar = torch.cat([box_idxs_pillar[:,:i,:,:], new_tensor[:,:,:,:], box_idxs_pillar[:,i:,:,:]])
                    #print(box_idxs_pillar.size())
                    #print(gt_boxes.size()[1])
                    
                    """
                    dict_seg.append(box_idxs_pillar)
                    #print(dict_seg[0].size())
                    #sys.exit()
                    #print(box_idxs_pillar.dtype)
                    dict_cls_num.append(limit.to(torch.int64))
                    #print(gt_boxes.size())

                """
                 end
                """
                
                targets_crr = torch.cat(dict_seg,dim=0)
                #print(batch_dict.keys())
                #print(batch_dict["spatial_features_2d"].size())
                #print(batch_dict["spatial_features"].size())
                spatial_features = batch_dict["spatial_features_2d"]
                pred_seg = self.segmentation_head(spatial_features)
                targets = batch_dict['one_hot']
                
                targets = (targets.bool() | targets_crr.bool()).to(torch.float32)                #sys.exit()
                #print(pred_seg.size())
                #print(targets.size())
                
                loss_seg = F.binary_cross_entropy_with_logits(pred_seg,targets,reduction='mean')
                #print(loss_seg)
                #sys.exit()
            
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss + loss_seg
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
