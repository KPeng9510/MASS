from .detector3d_template import Detector3DTemplate
from .segmentation_head import FCNMaskHead
import sys
from .erfnet import Net
import os
import torch.nn.functional as F
import torch
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from PIL import Image
import numpy as np
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
        #sys.exit()
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
                #print(batch_dict.keys())
                #sys.exit()
                points_mean = batch_dict["points_coor"]
                #print(points_mean.size())
                gt_boxes = batch_dict["gt_boxes"]
                #batch,c,h,w = points_mean.size()
                batch,c,h,w=2,3,512,512
                dict_seg = []
                dict_cls_num = []
                label_b = batch_dict["labels_seg"]

                
                for i in range(gt_boxes.size()[0]):
                    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
                    #print(points_mean.size())
                    points = points_mean
                    #print(points.size())
                    #sys.exit()
                    #print(points[:100,:])
                    #sys.exit()
                    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points.unsqueeze(dim=0).float().cuda(),
                    gt_boxes[i,:,:7].unsqueeze(dim=0).float().cuda()
                    ).long().squeeze(dim=0)
                    label = label_b[i].flatten()
                    gt_boxes_indx = gt_boxes[i,:,-1]
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
                    gt_boxes_indx = torch.cat([torch.Tensor([0]).cuda(),gt_boxes_indx],dim=0)
                    box_idxs_of_pts +=1
                    #print(gt_boxes_indx)
                    # = target_cr != 0
                    nonzero_mask = (label ==0)
                    label[nonzero_mask] = gt_boxes_indx[box_idxs_of_pts.long()][nonzero_mask]
                    nonzero_mask_2 = gt_boxes_indx[box_idxs_of_pts.long()] != 0
                    label[nonzero_mask_2] = gt_boxes_indx[box_idxs_of_pts.long()][nonzero_mask_2]
                    #print(gt_boxes_indx[box_idxs_of_pts.long()][nonzero_mask_2])
                    #print(label)
                    #sys.exit()
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
                    target_cr = label
                    limit = torch.max(target_cr)
                    
                    #sys.exit()
                    #print(limit)
                    #print(limit)
                    
                    #print(target_cr.size())
                    target_cr = label.view(1,1,512,512)
                    #print(torch.max(target_cr)[0])
                    #sys.exit()
                    #target_cr = torch.cat([target_cr, target_cr, target_cr], dim=-1)
                    #print(target_cr.size())
                    #im = Image.fromarray((target_cr.int().cpu().numpy()*10), 'RGB')
                    #f=open("/mrtstorage/users/kpeng/label.bin",'wb')
                    #f.write(label.view(h,w).cpu().numpy().astype(np.float32).tobytes())
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
                    dict_seg.append(target_cr)
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
                targets_crr = torch.cat(dict_seg,dim=0)
                #sys.exit()
                #print(batch_dict.keys())
                #print(batch_dict["spatial_features_2d"].size())
                #print(batch_dict["spatial_features"].size())
                spatial_features = batch_dict["spatial_features_2d"]
                pred_seg = self.segmentation_head(spatial_features)
                #print(pred_seg.size())
                #label = (np.argmax(pred_seg[0].cpu().numpy(), axis=0)).astype(np.float32).tobytes()
                #f=open("/mrtstorage/users/kpeng/labels.bin",'wb')
                #f.write(label)
                #f.close()
                #sys.exit()

                #targets = batch_dict['one_hot']
                #tar = torch.argmax(batch_dict['one_hot'],dim=1)
                #pred = torch.argmax(pred_seg, dim=1)
                #targets = (targets.bool() | targets_crr.bool()).to(torch.float32)
                targets = targets_crr
                target = targets
                #target = torch.argmax(targets, dim=1) #from 0 to 15
                nozero_mask = target != 0
                
                target = one_hot((target[nozero_mask]-1).long().unsqueeze(-1).unsqueeze(0).unsqueeze(0), 15)
                #print(pred_seg.size())
                pred = torch.argmax(pred_seg, dim=1).unsqueeze(1)
                pred = one_hot((pred[nozero_mask]).long().unsqueeze(-1).unsqueeze(0).unsqueeze(0),15)
                #sys.exit()
                #print(pred_seg.size())
                #print(targets.size())
                
                loss_seg = F.binary_cross_entropy_with_logits(pred,target,reduction='mean')
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
            loss, tb_dict, disp_dict = self.get_training_loss()
            #pred_boxes = batch_dict["batch_box_preds"]
            
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
