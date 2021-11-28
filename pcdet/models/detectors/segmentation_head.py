#import mmcv
import numpy as np
#import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import ConvModule
#from mmdet.core import mask_cross_entropy, mask_target
def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]

class FCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 in_channels=384,
                 conv_kernel_size=3,
                 conv_out_channels=384,
                 upsample_method='deconv',
                 upsample_ratio=4,
                 num_classes=15,
                 class_agnostic=False,
                 normalize=None):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        #self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels
                           if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    
                    padding=padding,
                    normalize=normalize,
                    bias=self.with_bias))
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                self.conv_out_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        loss_mask = mask_cross_entropy(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss
