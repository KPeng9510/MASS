from .detector3d_template import Detector3DTemplate
from .unet.unet import UNet
from .segmentation_head import FCNMaskHead
import sys
from .erfnet import Net
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from PIL import Image
import numpy as np
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size()[0], logit.size()[1], logit.size()[3])
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.unsqueeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size()[0], num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
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
def one_hot_1d(data, num_classes):
     n_values = num_classes
     n_values = torch.eye(n_values)[data]
     return n_values

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        #self.segmentation_head = FCNMaskHead()
        self.segmentation_head = UNet(64,13)
        self.focal_loss = FocalLoss()
    def forward(self, batch_dict):
        module_index = 0
        #print(batch_dict.keys())
        #sys.exit()
        
        for cur_module in self.module_list[:2]:
            module_index += 1
            batch_dict = cur_module(batch_dict)
            #print(batch_dict.keys())
            #torch.cuda.empty_cache()
            #points_mean = batch_dict['pointsmean']
            #batch_size,h,w = points_mean.size()
            #print(batch_dict["gt_boxes"][0,:,-1])
            if module_index == 2:
                """
                  encode bbox
                """
                #print(batch_dict.keys())
                #sys.exit()
                points_mean = batch_dict["points_coor"]
                #print(points_mean.size())
                #gt_boxes = batch_dict["gt_boxes"]
                #batch,c,h,w = points_mean.size()
                batch,c,h,w=2,3,1000,500
                dict_seg = []
                dict_cls_num = []
                label_b = batch_dict["labels_seg"]

                
                for i in range(1):
                    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
                    #print(points_mean.size())
                    #points = points_mean
                    #print(points.size())
                    #sys.exit()
                    #print(points[:100,:])
                    #sys.exit()
                    """
                    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points.unsqueeze(dim=0).float().cuda(),
                    gt_boxes[i,:,:7].unsqueeze(dim=0).float().cuda()
                    ).long().squeeze(dim=0)
                    label = label_b[i].flatten()
                    gt_boxes_indx = gt_boxes[i,:,-1]
                    """
                    #print(label)
                    #sys.exit()
                    """
                    if i == 1:
                        sys.exit()
                    """
                    #nonzero_number = torch.sum(nonzero_mask.float())
                    #print(gt_boxes_indx)
                    #print(box_idxs_of_pts.max())
                    #sys.exit()
                    #print(nonzero_number)
                    #print(gt_boxes_indx.size())
                    #if i == 1:
                    #    sys.exit()
                    #gt_boxes_indx = gt_boxes_indx[:nonzero_number.int()]
                    """
                    gt_boxes_indx = torch.cat([torch.Tensor([0]).cuda(),gt_boxes_indx],dim=0)
                    box_idxs_of_pts +=1
                    #print(box_idxs_of_pts)
                    #sys.exit()
                    # = target_cr != 0
                    #nonzero_mask = (label ==0)
                    #label[nonzero_mask] = gt_boxes_indx[box_idxs_of_pts.long()][nonzero_mask]
                    nonzero_mask_2 = gt_boxes_indx[box_idxs_of_pts.long()] != 0
                    label[nonzero_mask_2] = gt_boxes_indx[box_idxs_of_pts.long()][nonzero_mask_2]
                    #print(gt_boxes_indx[box_idxs_of_pts.long()][nonzero_mask_2])
                    #print(label)
                    #sys.exit()
                    """
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
                    #target_cr = label
                    #limit = torch.max(target_cr)
                    
                    #sys.exit()
                    #print(limit)
                    #print(limit)
                    
                    #print(target_cr.size())
                    #target_cr = label_b
                    #print(torch.max(target_cr)[0])
                    #sys.exit()
                    #target_cr = torch.cat([target_cr, target_cr, target_cr], dim=-1)
                    #print(target_cr.size())
                    #im = Image.fromarray((target_cr.int().cpu().numpy()*10), 'RGB')
                    #f=open("/mrtstorage/users/kpeng/label.bin",'wb')
                    #f.write(label_b[0].view(h,w).cpu().numpy().astype(np.float32).tobytes())
                    #f.close()
                    #sys.exit()
                    #target_label = torch.zeros([1,1,h,16], dtype=target_cr.dtype, device = target_cr.device)
                    #for i in range(16):
                    #    target_label[:,:,:,i] = i
                    #target_cr = torch.cat([target_cr,target_label],dim=-1)
                    #box_idxs_pillar = one_hot(target_cr.to(torch.int64), 16)
                    #box_idxs_pillar = box_idxs_pillar[:,:,:,:w]
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
                    dict_seg.append(label_b.unsqueeze(0))
                    #print(dict_seg[0].size())
                    #sys.exit()
                    #print(box_idxs_pillar.dtype)
                    #dict_cls_num.append(limit.to(torch.int64))
                    #print(gt_boxes.size())

                """
                 end
                """
                #print(dict_seg[0].size())
                #im = Image.fromarray(dict_seg[0].view([512,512,1]).cpu().numpy()*10)
                #im.save("/mrtstorage/users/kpeng/target.jpg")
                #print(dict_seg[0])
                targets_crr = torch.cat(dict_seg,dim=0).view(2,1,500,1000)
                #print(targets_crr[0])
                #sys.exit()
                #print(batch_dict.keys())
                #print(batch_dict["spatial_features_2d"].size())
                #print(batch_dict["spatial_features"].size())
                spatial_features = batch_dict["spatial_features"]
                pred = self.segmentation_head(spatial_features)
                
                #print(pred.size())
                #sys.exit()
                #print(targets_crr)
                
                #label = torch.argmax(pred[0].unsqueeze(0),dim=1).flatten().cpu().numpy().astype(np.float32).tobytes()
                #f=open("/mrtstorage/users/kpeng/labe.bin",'wb')
                #f.write(label)
                #f.close()
                #sys.exit()

                #targets = batch_dict['one_hot']
                #tar = torch.argmax(batch_dict['one_hot'],dim=1)
                #pred = torch.argmax(pred_seg, dim=1)
                #targets = (targets.bool() | targets_crr.bool()).to(torch.float32)
                targets_crr = targets_crr.contiguous().view(2,1,500,1000)
                
                #target = torch.argmax(targets, dim=1) #from 0 to 15
                nozero_mask = targets_crr != 0
                targets_crr = torch.clamp(targets_crr[nozero_mask],1,13)
                #print(target[nozero_mask])
                targets_crr = one_hot_1d((targets_crr-1).long(), 13).unsqueeze(0).permute(0,2,1).cuda()
                #print(target[:,-100:].size())
                #pred = torch.argmax(pred_seg, dim=1).unsqueeze(1)
                #print(target[nozero_mask])
                #sys.exit()
                #pred = one_hot_1d((pred[nozero_mask]).long(),15)
                #sys.exit()
                #print(pred_seg.size())
                #print(target.size())
                #sys.exit()
                pred = pred.permute(0,2,3,1).unsqueeze(1)[nozero_mask].squeeze().unsqueeze(0).permute(0,2,1)
                #print(pred.size())
                #print(targets_crr.size())
                #sys.exit()
                #pred = torch.argmax(pred,dim=1).permute(1,0)
                #targets_crr = torch.argmax(targets_crr,dim=1).permute(1,0)
                #print(pred.size())
                #print(targets_crr.size())
                #sys.exit()
                object_list = {1,3,4}
                for obj in object_list:
                    if obj == 1:
                        mask_obj = targets_crr == obj
                    else:
                        mask_obj = mask_obj | (targets_crr == obj)
                weight = torch.ones_like(targets_crr)
                mask_person = targets_crr == 2
                weight[mask_obj]==5
                weight[mask_person]==8
                loss_seg = F.binary_cross_entropy_with_logits(pred,targets_crr,reduction='mean',weight=weight)
                #loss_seg = self.focal_loss(pred.unsqueeze(1),targets_crr.unsqueeze(1))
                #print(loss_seg)
                #sys.exit()
        """
           code for geomertic consistency
        """
        #pred_dict,_=self.post_processing(batch_dict)
        
        #pred_boxs = pred_dict
        #positive_mask = pred_cls >= 1
        #print(pred_cls.size())
        #print(pred_boxs[0]["pred_boxes"].size())
        #print(pred_labels[0]["pred_labels"].size())
        #print(positive_mask.size())
        #p_box = pred_boxs[positive_mask]
        #pred_boxes = batch_dict[car_mask]
        #print(p_box)
        #sys.exit()
        #pred_dict,_=self.post_processing(batch_dict)
        #print(batch_dict.keys())
        if self.training:
            #loss, tb_dict, disp_dict = self.get_training_loss()
            #pred_boxes = batch_dict["batch_box_preds"]
            
            ret_dict = {
                'loss': loss_seg
            }
            disp_dict={}
            tb_dict={}
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
