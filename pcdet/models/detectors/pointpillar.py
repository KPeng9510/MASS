from .detector3d_template import Detector3DTemplate
from .unet.unet import UNet, SimplifiedUNet
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
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
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
        # self.segmentation_head = UNet(64, 12)
        # for attentional fusion
        # self.segmentation_head = SimplifiedUNet(64, 12)
        # for concat fusion
        self.segmentation_head = SimplifiedUNet(128, 12)
        self.focal_loss = FocalLoss()

    def forward(self, batch_dict):
        module_index = 0

        for cur_module in self.module_list[:2]:
            module_index += 1
            batch_dict = cur_module(batch_dict)
            if module_index == 2:
                # points_mean = batch_dict["points_coor"]
                dict_seg = []
                dict_cls_num = []
                label_b = batch_dict["labels_seg"]

                # batch, c, h, w = label_b.size()
                # targets_crr = label_b.view(batch, c, h, w)  # torch.cat(dict_seg,dim=0).view(batch,c,h,w)
                targets_crr = label_b
                spatial_features = batch_dict["spatial_features"]
                pred = self.segmentation_head(spatial_features)

                batch_dict['prediction'] = pred

                # label = torch.argmax(pred[0].unsqueeze(0),dim=1).flatten().cpu().numpy().astype(np.float32).tobytes()
                # f=open("/mrtstorage/users/kpeng/labe.bin",'wb')
                # f.write(label)
                # f.close()
                # sys.exit()

                # targets_crr = targets_crr.contiguous().view(batch, c, h, w)

        """
           code for geomertic consistency
        """
        if self.training:
            # targets_crr = targets_crr.contiguous().view(batch, c, h, w)
            nozero_mask = targets_crr != 0
            targets_crr = torch.clamp(targets_crr[nozero_mask], 1, 12)
            # ori_target = targets_crr
            targets_crr = one_hot_1d((targets_crr - 1).long(), 12).unsqueeze(0).permute(0, 2, 1).cuda()
            pred = pred.permute(0, 2, 3, 1).unsqueeze(1)[nozero_mask].squeeze().unsqueeze(0).permute(0, 2, 1)
            object_list = [0, 2, 3]
            # for obj in object_list:
            #    if obj == 1:
            #        mask_obj = ori_target == obj
            #    else:
            #        mask_obj = mask_obj | (targets_crr == obj)
            weight = torch.ones_like(targets_crr)
            # print(weight.size())
            # sys.exit()
            # mask_person = targets_crr == 1
            # weight[mask_obj]==5
            # weight[mask_person]==8
            # for dense gt TODO
            weight[:, 0, :] = 2  # weight 5 for other dynamic object
            weight[:, [1, 2, 3], :] = 7.5  # weight8 for pedestrain
            # for sparse
            # weight[:, 0, :] = 2  # weight 5 for vehicle
            # weight[:, [1, 2, 3], :] = 8  # weight8 for person, two wheel and rider
            loss_seg = F.binary_cross_entropy_with_logits(pred, targets_crr, reduction='mean', weight=weight)

            ret_dict = {
                'loss': loss_seg
            }
            disp_dict = {}
            tb_dict = {}
            return ret_dict, tb_dict, disp_dict
        else:
            return batch_dict
            # pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
